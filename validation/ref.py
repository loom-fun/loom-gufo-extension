#!/usr/bin/env python3
"""Greedy reference answers and behavior probes from any OpenAI server: ref.py BASE MODEL OUT.json"""
import json, sys, urllib.request
BASE, MODEL, OUT = sys.argv[1:4]
PROMPTS = ["What is the capital of France? Answer in one sentence.",
           "List the first five prime numbers, comma separated.",
           "Write a haiku about autumn leaves."]
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city.",
          "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}}, "required": ["city"]}}}]
def post(body):
    return json.load(urllib.request.urlopen(urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(),
                                                                   {"Content-Type": "application/json"}), timeout=900))
out = []
for p in PROMPTS:
    try:
        r = post({"model": MODEL, "messages": [{"role": "user", "content": p}], "max_tokens": 64, "temperature": 0,
                  "chat_template_kwargs": {"enable_thinking": False}})
        m = r["choices"][0]["message"]
        out.append({"content": m.get("content"), "reasoning": m.get("reasoning_content") or "", "tokens": r["usage"]["completion_tokens"]})
    except Exception as e:
        out.append({"error": str(e)})
probes = {}
try:
    r = post({"model": MODEL, "messages": [{"role": "user", "content": "What's the weather in Paris right now? Use the get_weather tool."}],
              "tools": TOOLS, "max_tokens": 2000, "temperature": 0})
    probes["tool_call"] = bool(r["choices"][0]["message"].get("tool_calls"))
except Exception as e:
    probes["tool_call_error"] = str(e)[:200]
try:
    r = post({"model": MODEL, "messages": [{"role": "user", "content": "What is 23*17? Think step by step, then give the number."}],
              "max_tokens": 3000, "temperature": 0, "chat_template_kwargs": {"enable_thinking": True}})
    m = r["choices"][0]["message"]
    probes["thinking_answer"] = "391" in (m.get("content") or "")
    probes["thinking_finish"] = r["choices"][0]["finish_reason"]
except Exception as e:
    probes["thinking_error"] = str(e)[:200]
json.dump({"answers": out, "probes": probes} if False else out + [{"probes": probes}], open(OUT, "w"), indent=1)
print(json.dumps(probes))
