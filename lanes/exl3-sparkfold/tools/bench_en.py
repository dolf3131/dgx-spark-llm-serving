#!/usr/bin/env python3
"""Single-stream decode tok/s of an OpenAI-compatible server (English prompts, greedy), isolating decode from prefill via streaming."""
import json, os, sys, time, urllib.request

URL = "http://127.0.0.1:8888/v1/chat/completions"
# Served model name varies by container (qwen3.6, nemotron, ling3, ...); the port does
# not, since only one model fits in memory at a time.
MODEL = os.environ.get("BENCH_MODEL", "qwen3.8-flash-next")

PROMPTS = {
    "LRU   ": "Implement a thread-safe LRU cache in Python with get and put in O(1). Include type hints, a docstring, and a short explanation of the eviction policy.",
    "CODE-2": "Write a Python class for a binary search tree with insert, delete, and in-order traversal. Full type hints.",
    "PROSE ": "Explain the causes of the French Revolution in flowing prose. No lists, no code.",
    "CODE-3": "Refactor this into idiomatic Rust: a function that reads a file line by line and counts word frequencies into a HashMap.",
}


def run(label, prompt, max_tokens=600):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})

    t0 = time.perf_counter()
    t_first = None
    t_last = None
    ntok = 0
    usage = None
    with urllib.request.urlopen(req) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            d = json.loads(payload)
            if d.get("usage"):
                usage = d["usage"]
            # Count every streamed chunk: while the model is inside a thinking
            # block the parser emits neither content nor reasoning_content, but
            # tokens are still being produced.
            if d.get("choices"):
                now = time.perf_counter()
                if t_first is None:
                    t_first = now
                t_last = now
                ntok += 1

    if t_first is None:
        print(f"{label} -> no tokens received")
        return
    ttft = t_first - t0
    decode_s = (t_last - t_first) or 1e-9
    gen = usage["completion_tokens"] if usage else ntok
    # With speculative decoding one SSE chunk can carry several tokens, so rate
    # must come from the tokenizer's count (usage), not the chunk count.
    rate = gen / decode_s
    print(f"{label} -> decode {rate:6.2f} tok/s | TTFT {ttft*1000:6.0f} ms | "
          f"gen {gen} tok in {decode_s:5.2f}s | chunks {ntok} ({gen/max(ntok,1):.2f} tok/chunk)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "warmup":
        print("warmup...")
        run("WARM  ", "Say hello.", max_tokens=8)
    else:
        for label, p in PROMPTS.items():
            run(label, p)
