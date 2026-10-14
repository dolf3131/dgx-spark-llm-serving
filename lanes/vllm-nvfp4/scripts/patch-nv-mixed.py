"""Serve nvidia/Qwen3.8-Flash-Next-NVFP4 (ModelOpt MIXED_PRECISION) on this image.

NVIDIA's card names both gaps this closes:

    "Serving without MTP requires vLLM commit d4d703ca... or later. MTP speculative
     decoding additionally requires vLLM PR #55513 until that fix is merged upstream."

Our image is pinned at 0.1.dev20073+g8e685d198, older than both. Part A below is our own
fix for the first gap; part B is a port of PR #55513 (still open upstream, 5 files, of
which 3 are production code and 2 are tests we skip).

The checkpoint declares itself per layer in config.json's quantization_config:

    model.language_model.layers.N.mlp.experts            -> NVFP4      gs=16
    model.language_model.layers.1.ple...ngram_embedding  -> FP8
    mtp.layers.0.mlp.experts                             -> FP8_PB_WO  gs=128

-----------------------------------------------------------------------------
A. ple_layer.py -- let the PLE load FP8 under a MIXED_PRECISION config
-----------------------------------------------------------------------------
`_get_ple_embedding_quant_method` gates on the NATIVE Fp8Config
(vllm...quantization.fp8). MIXED_PRECISION arrives as ModelOptMixedPrecisionConfig, so
the isinstance check fails, the PLE is built unquantized, and the 128 F8_E4M3 shards in
model-fp8-mtp-ple.safetensors have nowhere to load. Same gate that rejects RadixArk's
fp8 PLE. Nothing has to be invented: the mixed config already resolves this prefix to
"FP8" via `_resolve_quant_algo` and already carries a real ModelOptFp8Config.

Layout verified against the checkpoint's own tensor header before writing (range request
on model-fp8-mtp-ple.safetensors), because the embedding method only accepts one shape:

    ngram_embedding.shard_N.weight   F8_E4M3  (2500012, 160)  x128
    ngram_embedding.weight_scale     BF16     (1,)            <- per-table scalar

-----------------------------------------------------------------------------
B. PR #55513 -- block-FP8 MoE for the MTP drafter
-----------------------------------------------------------------------------
Without it the boot dies at the very end of weight loading with

    AttributeError: Layer mtp.layers.48.mlp.experts has no parameter
    'w2_weight_scale_inv' for checkpoint weight
    'mtp.layers.48.mlp.experts.0.down_proj.weight_scale_inv'

Two independent causes, and the PR fixes both:

  B1. RoutedExperts with quant_algo FP8_PB_WO fell through to ModelOptFp8MoEMethod,
      which registers PerTensorScaleParameter `w13/w2_weight_scale` -- one scalar per
      expert. The checkpoint ships 2D block scales (`weight_scale_inv`, e.g. (20, 5) for
      down_proj at group_size 128). The native Fp8MoEMethod already handles exactly this:
      its ctor sets `weight_scale_name = "weight_scale_inv"` when `block_quant`. So route
      block-FP8 experts there instead. This image already knows FP8_PB_WO for LINEAR
      layers (ModelOptFp8PBWO...), just not for MoE.

  B2. The layer index in the error is 48, but the checkpoint says mtp.layers.0. The
      draft model is built standalone, so its layer indices are offset by
      mtp_start_layer_idx. `_remap_ignored_layers` already exists and is applied to
      ignored_layers/exclude_modules -- but not to `quantized_layers`, so every MTP entry
      missed its lookup. Apply the same remap.

`has_blocked_weights` from the PR is skipped: this image's mixed config has no such
method, so there is nothing to widen.

Usage:  python3 patch_nv_mixed.py
"""

import glob
import sys


def sub(text: str, old: str, new: str, what: str) -> str:
    n = text.count(old)
    if n != 1:
        print(f"patch_nv_mixed: FAILED -- anchor {what!r} matched {n} times", file=sys.stderr)
        raise SystemExit(1)
    return text.replace(old, new)


def one(pattern: str) -> str:
    hits = glob.glob(pattern)
    if not hits:
        print(f"patch_nv_mixed: FAILED -- {pattern} not found", file=sys.stderr)
        raise SystemExit(1)
    return hits[0]


BASE = "/usr/local/lib/python3.*/dist-packages/vllm"

# --------------------------------------------------------------------------- #
# A. PLE FP8 under MIXED_PRECISION
# --------------------------------------------------------------------------- #
ple_path = one(f"{BASE}/models/qwen3_8_flash_next/nvidia/ple_layer.py")
s = open(ple_path).read()
s = sub(
    s,
    '''    """Select global-scale FP8 only for quantized PLE checkpoint shards."""

    if not isinstance(quant_config, Fp8Config):
        return None
''',
    '''    """Select global-scale FP8 only for quantized PLE checkpoint shards."""

    # ModelOpt MIXED_PRECISION checkpoints (nvidia/Qwen3.8-Flash-Next-NVFP4) arrive as
    # ModelOptMixedPrecisionConfig, not the native Fp8Config this gate was written for.
    # The mixed config already resolves this prefix to FP8 via quantized_layers and holds
    # a real ModelOptFp8Config, so honour it instead of silently building an unquantized
    # PLE that cannot load the checkpoint's F8_E4M3 shards.
    try:
        from vllm.model_executor.layers.quantization.modelopt import (
            ModelOptMixedPrecisionConfig,
        )
    except ImportError:  # modelopt unavailable in this build
        ModelOptMixedPrecisionConfig = ()
    if ModelOptMixedPrecisionConfig and isinstance(
        quant_config, ModelOptMixedPrecisionConfig
    ):
        if quant_config.is_layer_excluded(prefix):
            return None
        if quant_config._resolve_quant_algo(prefix) != "FP8":
            return None
        return Qwen3_8FlashNextPLEFp8EmbeddingMethod()

    if not isinstance(quant_config, Fp8Config):
        return None
''',
    "A: PLE mixed-precision FP8 route",
)
open(ple_path, "w").write(s)
print(f"patch_nv_mixed: A  patched {ple_path}")

# --------------------------------------------------------------------------- #
# B1. PR #55513 -- block-FP8 MoE dispatch
# --------------------------------------------------------------------------- #
mo_path = one(f"{BASE}/model_executor/layers/quantization/modelopt.py")
s = open(mo_path).read()
s = sub(
    s,
    """        if isinstance(layer, RoutedExperts):
            if quant_algo == "FP8":
                return ModelOptFp8MoEMethod(""",
    """        if isinstance(layer, RoutedExperts):
            # PR #55513: block-scaled FP8 experts. ModelOpt's canonical name is
            # FP8_PB_WO; early composed Qwen3.8-Flash-Next checkpoints used
            # FP8_BLOCK_SCALES for the same layout. ModelOptFp8MoEMethod below only
            # registers per-tensor scales, so it cannot load `weight_scale_inv`.
            # The native Fp8MoEMethod sets weight_scale_name="weight_scale_inv"
            # whenever weight_block_size is set, which is exactly this layout.
            if quant_algo in ("FP8_PB_WO", "FP8_BLOCK_SCALES"):
                from vllm.model_executor.layers.quantization.fp8 import (
                    Fp8Config as _NativeFp8Config,
                )
                from vllm.model_executor.layers.quantization.fp8 import (
                    Fp8MoEMethod as _NativeFp8MoEMethod,
                )

                _sizes = {
                    int(info.get("group_size", 128))
                    for info in self.quantized_layers.values()
                    if info.get("quant_algo", "").upper()
                    in ("FP8_PB_WO", "FP8_BLOCK_SCALES")
                }
                if len(_sizes) > 1:
                    raise ValueError(
                        "MIXED_PRECISION currently requires all block-FP8 MoE "
                        f"layers to use one group_size, got {sorted(_sizes)}."
                    )
                _bs = next(iter(_sizes), 128)
                return _NativeFp8MoEMethod(
                    _NativeFp8Config(
                        is_checkpoint_fp8_serialized=True,
                        activation_scheme="dynamic",
                        weight_block_size=[_bs, _bs],
                    ),
                    layer,
                )
            if quant_algo == "FP8":
                return ModelOptFp8MoEMethod(""",
    "B1: block-FP8 MoE dispatch",
)
open(mo_path, "w").write(s)
print(f"patch_nv_mixed: B1 patched {mo_path}")

# --------------------------------------------------------------------------- #
# B2. PR #55513 -- remap quantized_layers for the standalone draft model
# --------------------------------------------------------------------------- #
mtp_path = one(f"{BASE}/models/qwen3_8_flash_next/nvidia/mtp.py")
s = open(mtp_path).read()
s = sub(
    s,
    """        exclude_modules = getattr(draft_quant_config, "exclude_modules", None)
        if exclude_modules:""",
    """        # PR #55513: quantized_layers is keyed by CHECKPOINT indices (mtp.layers.0),
        # but the draft model is standalone and its layers are offset by
        # mtp_start_layer_idx. Without this every MTP entry misses its lookup and the
        # experts fall back to the wrong quant method.
        quantized_layers = getattr(draft_quant_config, "quantized_layers", None)
        if quantized_layers:
            setattr(  # noqa: B010
                draft_quant_config,
                "quantized_layers",
                {
                    _remap_ignored_layers([name], mtp_start_layer_idx)[0]: info
                    for name, info in quantized_layers.items()
                },
            )
        exclude_modules = getattr(draft_quant_config, "exclude_modules", None)
        if exclude_modules:""",
    "B2: remap quantized_layers",
)
open(mtp_path, "w").write(s)
print(f"patch_nv_mixed: B2 patched {mtp_path}")
print("patch_nv_mixed: OK")
