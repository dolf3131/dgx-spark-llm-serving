#!/usr/bin/env python3
"""Two long conversations taking turns (the 10-10 Cursor/VS Code shape): does each next turn resume from its own kept state?
A ~140k and B ~100k tokens, each turn appends the reply and one short question, thinking off, max_tokens 16.
usage: two_conv.py <rounds> [A_tokens B_tokens]   prints one line per request: tokens, cached, prefill from /health, wall."""
import glob, json, os, sys, time, urllib.request
URL = "http://127.0.0.1:8888"
ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 4
TARGET = {"A": int(sys.argv[2]) if len(sys.argv) > 2 else 140000, "B": int(sys.argv[3]) if len(sys.argv) > 3 else 100000}
if len(sys.argv) > 4: TARGET["C"] = int(sys.argv[4])
NAMES = tuple(TARGET)
# the prompts are made of public text: the engine tree's own sources and docs (TF_TREE, the patched tree the server runs)
TREE = os.path.expanduser(os.environ.get("TF_TREE", "."))
files = sorted(glob.glob(os.path.join(TREE, "src/**/*.py"), recursive=True)) + sorted(glob.glob(os.path.join(TREE, "**/*.md"), recursive=True))
corpus = "\n\n".join(f"### {p}\n" + open(p, encoding="utf-8", errors="replace").read() for p in files if "draft_vocab" not in p)
print(f"corpus {len(corpus):,} chars from {len(files)} files", flush=True)
def post(path, body, timeout=1200):
    return json.load(urllib.request.urlopen(urllib.request.Request(URL + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=timeout))
def health(): return json.load(urllib.request.urlopen(URL + "/health", timeout=10))
def chat(messages):
    h0 = health(); t0 = time.perf_counter()
    d = post("/v1/chat/completions", {"model": "qwen3.8-flash-next", "messages": messages, "max_tokens": 16, "temperature": 0,
                                       "chat_template_kwargs": {"enable_thinking": False}})
    wall = time.perf_counter() - t0; h1 = health()
    u = d["usage"]; cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    return d["choices"][0]["message"].get("content") or "", u["prompt_tokens"], cached, h1["prefill_seconds_total"] - h0["prefill_seconds_total"], wall
# calibrate chars per token on a 4k-char probe (also warms nothing: the text is not reused)
_, p0, _, _, _ = chat([{"role": "user", "content": corpus[-4000:] + "\nSay ok."}])
cpt = 4000 / max(1, p0 - 20); print(f"calibration: {cpt:.2f} chars/token", flush=True)
convs, pos = {}, 0
for name in NAMES:
    n = int(TARGET[name] * cpt); text = corpus[pos:pos + n]; pos += n
    convs[name] = [{"role": "system", "content": "You answer with one word."},
                   {"role": "user", "content": "Document:\n" + text + "\n\nQuestion 0: reply ok."}]
print(f"{'conv':<5}{'turn':<5}{'prompt':>8}{'cached':>8}{'prefill_s':>10}{'wall_s':>8}", flush=True)
for turn in range(ROUNDS + 1):
    for name in NAMES:
        reply, ptok, cached, pre, wall = chat(convs[name])
        print(f"{name:<5}{turn:<5}{ptok:>8}{cached:>8}{pre:>10.1f}{wall:>8.1f}", flush=True)
        convs[name] += [{"role": "assistant", "content": reply or "ok"}, {"role": "user", "content": f"Question {turn + 1}: reply ok."}]
