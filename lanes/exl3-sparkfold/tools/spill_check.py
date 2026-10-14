#!/usr/bin/env python3
"""Real-model check of the NVMe spill: A, B, C (long documents) are read cold, then A, B, C again as the identical request and as follow-ups.
usage: spill_check.py run <A_tokens> <B_tokens> <C_tokens> [state.json]   |   spill_check.py again state.json   (after a server restart)
Exactness: an identical request answered after the conversation was spilled and read back must give the SAME greedy reply as its cold read."""
import glob, hashlib, json, os, sys, time, urllib.request
URL = "http://127.0.0.1:8888"
# the prompts are made of public text: the engine tree's own sources and docs (TF_TREE, the patched tree the server runs)
TREE = os.path.expanduser(os.environ.get("TF_TREE", "."))
files = sorted(glob.glob(os.path.join(TREE, "src/**/*.py"), recursive=True)) + sorted(glob.glob(os.path.join(TREE, "**/*.md"), recursive=True))
corpus = "\n\n".join(f"### {p}\n" + open(p, encoding="utf-8", errors="replace").read() for p in files if "draft_vocab" not in p)
def post(body, timeout=1800):
    return json.load(urllib.request.urlopen(urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=timeout))
def health(): return json.load(urllib.request.urlopen(URL + "/health", timeout=10))
def ask(messages, n=32):
    h0 = health(); t0 = time.perf_counter()
    d = post({"model": "qwen3.8-flash-next", "messages": messages, "max_tokens": n, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
    wall = time.perf_counter() - t0; h1 = health()
    u = d["usage"]; cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    return d["choices"][0]["message"].get("content") or "", u["prompt_tokens"], cached, h1["prefill_seconds_total"] - h0["prefill_seconds_total"], wall
def build(sizes, cpt):
    pos, convs = 0, {}
    for name, tok in zip("ABC", sizes):
        n = int(tok * cpt); convs[name] = [{"role": "system", "content": "You answer briefly."},
            {"role": "user", "content": f"Document {name}:\n" + corpus[pos:pos + n] + "\n\nSummarize the first file's purpose in one sentence."}]; pos += n
    return convs
def row(label, name, r):
    reply, p, c, pre, wall = r
    print(f"{label:<22}{name:<3}{p:>8}{c:>8}{pre:>9.1f}{wall:>8.1f}  {hashlib.sha1(reply.encode()).hexdigest()[:8]}  {reply[:50]!r}", flush=True)
print(f"{'step':<22}{'cv':<3}{'prompt':>8}{'cached':>8}{'prefill':>9}{'wall':>8}  reply")
if sys.argv[1] == "run":
    sizes = [int(x) for x in sys.argv[2:5]]; state = sys.argv[5] if len(sys.argv) > 5 else "spill_state.json"
    _, p0, _, _, _ = ask([{"role": "user", "content": corpus[-4000:] + "\nSay ok."}])
    cpt = 4000 / max(1, p0 - 20); convs = build(sizes, cpt); cold = {}
    for name in "ABC":
        r = ask(convs[name]); cold[name] = r[0]; row("cold", name, r)
    bad = 0
    for rnd in (1, 2):
        for name in "ABC":
            r = ask(convs[name]); row(f"identical again #{rnd}", name, r)
            if r[0] != cold[name]: bad += 1; print("   !!! REPLY DIFFERS FROM THE COLD READ", flush=True)
    for name in "ABC":                                           # follow-ups: the reply and one more question
        convs[name] += [{"role": "assistant", "content": cold[name]}, {"role": "user", "content": "And its second file?"}]
        row("follow-up", name, ask(convs[name]))
    json.dump({"sizes": sizes, "cpt": cpt, "cold": cold, "A": convs["A"][:2]}, open(state, "w"))
    print(f"identical-request replies that differ from their cold read: {bad}")
else:
    st = json.load(open(sys.argv[2]))
    follow = st["A"] + [{"role": "assistant", "content": st["cold"]["A"]}, {"role": "user", "content": "And its second file?"}]
    r = ask(follow); row("follow-up after restart", "A", r)             # the conversation moves forward from its newest stored state
    print("reply hash (compare with the follow-up row of the run before the restart):", hashlib.sha1(r[0].encode()).hexdigest()[:8])
