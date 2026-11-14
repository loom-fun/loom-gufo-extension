#!/usr/bin/env python3
"""One long greedy request; prints content and cache timings: disk_test.py BASE"""
import json, sys, urllib.request
BASE = sys.argv[1]
MODEL = json.load(urllib.request.urlopen(BASE + "/v1/models"))["data"][0]["id"]
doc = " ".join(f"Fact {i}: the river number {i} flows into lake {i * 7 % 13} near the town of Alder{i % 5}." for i in range(220))
body = {"model": MODEL, "temperature": 0, "max_tokens": 40, "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "system", "content": "You answer questions about the document.\n" + doc},
                     {"role": "user", "content": "Which lake does river 17 flow into? Answer briefly."}]}
r = json.load(urllib.request.urlopen(urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600))
t = r.get("timings", {})
print(json.dumps({"content": r["choices"][0]["message"]["content"], "prompt": r["usage"]["prompt_tokens"], "cache_n": t.get("cache_n"),
                  "reuse": (t.get("component_cache") or {}).get("reuse")}))
