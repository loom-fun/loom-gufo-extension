#!/usr/bin/env python3
"""Haiku greedy answer alone vs while 3 other requests decode: haikuload.py BASE"""
import json, sys, urllib.request, concurrent.futures as cf, time
BASE = sys.argv[1]
MODEL = json.load(urllib.request.urlopen(BASE + "/v1/models"))["data"][0]["id"]
def ask(p, n=64):
    body = {"model": MODEL, "messages": [{"role": "user", "content": p}], "max_tokens": n, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    r = json.load(urllib.request.urlopen(urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600))
    return r["choices"][0]["message"].get("content")
H = "Write a haiku about autumn leaves."
alone = ask(H)
outs = set()
for delay in (0.0, 0.05, 0.15, 0.3):
    with cf.ThreadPoolExecutor(4) as ex:
        bg = [ex.submit(ask, f"Write a long story about {t}.", 300) for t in ("rivers", "mountains", "cities")]
        time.sleep(delay)
        outs.add(ex.submit(ask, H).result())
print(json.dumps({"alone": alone, "under_load_variants": sorted(outs), "all_same": outs == {alone}}))
