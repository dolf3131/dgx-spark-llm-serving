#!/usr/bin/env python3
"""Korean prose decode rate on the path Open WebUI uses: /v1/chat/completions with no
sampling overrides, so the server defaults (checkpoint config + top_k override) apply and
thinking is on. Same streaming measurement as bench_en.py; 600 tokens is a measurement
window, not a serving cap.

    ./bench_ko.py
"""
import os
import bench_en as b

b.MODEL = os.environ.get("BENCH_MODEL", "qwen3.8-flash-next")
PROMPTS = {
    "KO-HIST ": "조선 후기 실학이 등장한 배경과 그 의의를 자연스러운 문단으로 설명해 줘. 목록은 쓰지 마.",
    "KO-SCI  ": "광합성의 명반응과 암반응이 어떻게 연결되는지 고등학생이 이해할 수 있게 설명해 줘.",
    "KO-ADV  ": "30대 직장인이 퇴근 후 공부 습관을 만들려면 어떻게 해야 하는지 현실적인 조언을 해 줘.",
}


def run(label, prompt, max_tokens=600):
    # bench_en.run sends temperature 0; drop it so the server default applies.
    orig = b.json.dumps

    def dumps(obj, *a, **k):
        if isinstance(obj, dict) and "messages" in obj:
            obj = {k2: v for k2, v in obj.items() if k2 != "temperature"}
        return orig(obj, *a, **k)

    b.json.dumps = dumps
    try:
        b.run(label, prompt, max_tokens)
    finally:
        b.json.dumps = orig


if __name__ == "__main__":
    b.run("WARM    ", "Say hello.", max_tokens=8)
    for label, p in PROMPTS.items():
        run(label, p)
