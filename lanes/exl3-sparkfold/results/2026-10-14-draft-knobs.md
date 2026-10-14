# Draft-stop knobs, measured 2026-10-14

Does `TF_DRAFT_COST` (the expected-time draft stop, default 0.06 tokens/ms, tuned by SparkFold with its trained MTP head) or
`TF_JOINT_PRICE=1` (joint draft pricing for shared rounds) help when the engine drafts with the checkpoint's own MTP head? Both change speed
only; replies are unchanged.

Method: the deployed configuration restarted once per arm with one knob changed, server idle. `tools/bench_openai.py` from the TensorFold tree:
2 short English prompts (code and chat) at temperature 1.0 and 0, 128 tokens, 3 repetitions; `tools/bench_concurrent.py`: the same prompts at
1 and 3 concurrent streams, 96 tokens, 2 repetitions. One run per arm.

| arm | single stream, tok/s | 3 streams, aggregate | 3 streams, per stream | accepted / drafted |
|---|---|---|---|---|
| baseline (`TF_DRAFT_COST` 0.06) | 71.0 | 131.2 | 45.4 | 0.683 |
| `TF_DRAFT_COST=0.04` | 70.6 | 130.6 | 45.1 | 0.680 |
| `TF_DRAFT_COST=0.08` | 70.9 | 133.9 | 46.0 | 0.690 |
| `TF_JOINT_PRICE=1` | 71.0 | 124.9 | 43.3 | 0.686 |

Single-stream speed is flat to within 0.6%. `TF_DRAFT_COST=0.08` is +2% at 3 streams, inside the scatter we see from day to day on this server
(the same code measured 42.6 and 59.6 tok/s English decode on two days). `TF_JOINT_PRICE=1` is 5% lower at 3 streams. Conclusion for this
workload: no lever; the defaults stay.

Not covered: long contexts, Korean text, agent-shaped requests, 5 streams, repeated runs.
