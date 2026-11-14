#!/usr/bin/env python3
"""How a server handles 'thinking off': thinkoff.py BASE"""
import json, sys, urllib.request
BASE = sys.argv[1]
MODEL = json.load(urllib.request.urlopen(BASE + "/v1/models"))["data"][0]["id"]
def chat(**kw):
    body = {"model": MODEL, "messages": [{"role": "user", "content": "What is 23*17? Give only the number."}], "temperature": 0, "max_tokens": 300, **kw}
    try:
        r = json.load(urllib.request.urlopen(urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600))
    except urllib.error.HTTPError as e:
        return {"error": e.code, "body": e.read().decode()[:200]}
    m = r["choices"][0]["message"]
    return {"reasoning": len(m.get("reasoning_content") or ""), "content": (m.get("content") or "")[:60],
            "rt": r["usage"].get("completion_tokens_details", {}).get("reasoning_tokens"), "n": r["usage"]["completion_tokens"]}
for name, kw in [("default", {}), ("enable_thinking=false", {"chat_template_kwargs": {"enable_thinking": False}}),
                 ("reasoning_effort=none", {"reasoning_effort": "none"}), ("reasoning_effort=low", {"reasoning_effort": "low"})]:
    print(name, json.dumps(chat(**kw)))
