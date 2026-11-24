# Hy4-flash on one DGX Spark: EXL3 on busitdol

What one Spark serves here since 2026-11-19 (this build since 2026-11-24): Tencent's
[Hy4-flash](https://huggingface.co/tencent/Hy4-flash) from an EXL3 pack we quantized ourselves, on the
[busitdol](https://github.com/dolf3131/busitdol) engine (a fork of TensorFold by way of SparkFold), with Tencent's DFly2 drafter
retrained on the model's own answers. One request at a time by default and up to eight at once, a 1M-token context,
image input.

Measurements are dated. Read [Limits](#limits-and-open-items) before relying on any of them.

## TL;DR

| | |
|---|---|
| Hardware | 1x DGX Spark (GB10, 121 GiB unified memory) |
| Model | 220B in total (155.5B in the MoE trunk, 63.5B in Engram lookup tables), 11.3B active a token. 42 layers: 1 dense, 41 latent MoE (16 of 512 experts plus a shared one). MLA attention everywhere: 12 DSA layers (compressed indexer, top 2,048 keys) and 30 sliding-window layers (512). Engram tables at layers 1, 11, 21 and 31. A 5-layer DFly2 block drafter. |
| Pack | Ours, quantized with exllamav3 1.5.4 (details in [The pack](#the-pack)): attention K4, routed experts K3/K4, shared experts K4, latent projections and output head K6, a few small projections BF16 (int8 at load), Engram tables fp8, the retrained drafter K4. 72 GB of weights plus 79 GB of Engram tables, which are read from NVMe. Not published. |
| Engine | [busitdol](https://github.com/dolf3131/busitdol) `main` at `96cf97a`, mounted over the image `tensorfold-qwen38:v0.3.6.3-xgrammar` |
| Serving | [scripts/start.sh](scripts/start.sh): `--parallel 1 --context 1048576 --temperature 0.9 --top-k 10 --thinking --vision`; up to 8 concurrent streams; int8 latent KV cache; up to 5 drafts a round |
| Decode, one stream | 49.1 tok/s on eight real prompts (sampled, 2026-11-24); about 55 tok/s in real chat use (2026-11-24) |
| Decode, several streams | 78.2 tok/s in total with four at a time (2026-11-24); sgbench end to end, time to first token included: 51.4 / 64.2 / 85.7 tok/s at 1 / 2 / 4 streams (2026-11-23) |
| Prefill | about 1,580 tok/s at 8.4k and 1,600 tok/s at 28k prompt tokens, cold (2026-11-22) |
| Quality | against BF16 on 2,600 Korean tokens: KL 0.042, top-1 agreement 92.7%; tool-eval 95/100 (185/194 points) |
| Cold start | about 45 s with the extensions built and the caches warm; the first start compiles for several minutes |

Details of each number: [results/2026-11-24-hy4-flash.md](results/2026-11-24-hy4-flash.md).

## What is in this directory

| | |
|---|---|
| [scripts/start.sh](scripts/start.sh) | the `docker run` for this lane, every setting with a one-line note and its off value |
| [results/](results/) | dated measurements and how they were taken |

The engine code is in [busitdol](https://github.com/dolf3131/busitdol), not here.

## The pack

Built from Tencent's BF16 release in steps, each step measured against BF16 before the next:

1. **exllamav3 1.5.4**, its default calibration mix (250 rows of 2,048 tokens; it holds almost no Korean). All experts
   are activated during calibration; gate and up share one Hadamard rotation; each layer is quantized after the
   previous layers' quantized weights are in place. First pack: routed experts K4, everything else K6.
2. **Attention to K4** (MLA projections including `o_proj` and `linear_gate`).
3. **Routed experts' gate and up to K3** on the 22 layers that Tencent's AngelSlim Q3_K recipe puts at 3 bits
   (6-9, 12, 16, 18, 19, 21, 22, 24-26, 29-31, 34, 35, 37, 39-41, read from their GGUF header); down stays K4, and the
   other layers stay K4.
4. **Shared experts and the layer-0 MLP to K4.** The routed experts' latent projections and the output head stay K6.
5. **Kept in BF16 in the pack and turned to int8 at load:** `kv_a` (576 outputs, not a multiple of exllamav3's
   128-wide rotation), `kv_b`, the router, the hyper-connection mixes, and the indexer and Engram projections. The
   Engram tables are fp8 rows in `engram/`.
6. **The drafter:** Tencent's DFly2 weights, retrained on about 1.1M tokens of this pack's own answers, then fine-tuned
   on 1,000 answers with a total-variation loss, then its 39 linears quantized to K4. Drafts are only proposals: every
   token is checked by the target, so the drafter changes speed, not what greedy decoding returns.

KL against BF16 after each step (2,600 Korean tokens): 0.019 (step 1), 0.028 (2), 0.041 (2 + 3), 0.042 (+ 4).
The scripts that did this are research code tied to our containers and are not published yet.

## Reproduce

1. **Prerequisites.** A DGX Spark, Docker with the NVIDIA runtime, git, about 160 GB free on NVMe for the pack.
2. **The engine.** `git clone https://github.com/dolf3131/busitdol ~/busitdol`
3. **The image.** As in the [EXL3 Qwen lane](../exl3-sparkfold/README.md#reproduce), step 3: MiaAI-Lab's
   `tensorfold-qwen38:v0.3.6.3` plus `xgrammar`. It supplies PyTorch and the dependencies; the busitdol tree replaces the
   code inside it.
4. **The pack.** Not published (see [The pack](#the-pack)); `MODEL` must point at a pack with the same layout, including
   `engram/engram.json`.
5. **Start.**
   ```bash
   TF_SRC=~/busitdol MODEL=/models/<your pack> ./scripts/start.sh     # port 8888 on 127.0.0.1
   curl -s localhost:8888/health
   ```

## Limits and open items

- **Korean tokens.** Hy4's tokenizer has 1,291 tokens with Hangul in them (Qwen3.8: 6,793), so Korean answers take
  about 1.08 times Qwen3.8's tokens (1.25 times for pure Hangul). Korean answers sometimes contain Chinese characters;
  this lane has no language gate.
- **Two settings change outputs slightly** and are on: the target reads its output head through the 2-bit screen with
  int8 rescoring (its top token agrees with the full head's on 99.6-99.8% of real rows), and relaxed acceptance of drafts
  in sampled thinking and answer spans. `KNOBS="-e TF_HY4_TARGET_SCREEN=0 -e TF_HY4_RELAX_MARGIN=0 -e TF_HY4_RELAX_ANSWER_MARGIN=0"`
  turns both off.
- **Quality was measured two ways only:** KL on Korean text against BF16, and tool-eval. The tool-eval comparison with
  our earlier K6 pack (90/100) is not controlled: the engine changed between the two runs.
- **One lane at a time.** The pack, the KV pool and the image services of this box do not leave room for a second model.
- **Neither the pack nor the drafter is published.**
