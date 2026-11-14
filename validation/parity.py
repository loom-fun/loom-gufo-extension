#!/usr/bin/env python3
"""Behavior parity on loom-serve for checks that depend on the model: parity.py BASE MODEL OUT.json"""
import json, sys, urllib.request
BASE = sys.argv[1]
MODEL = json.load(urllib.request.urlopen(BASE + "/v1/models"))["data"][0]["id"]
OUT = sys.argv[2] if len(sys.argv) > 2 else "/dev/null"
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city.",
          "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}}, "required": ["city"]}}}]
def chat(messages, **kw):
    body = {"model": MODEL, "messages": messages, "temperature": 0, **kw}
    return json.load(urllib.request.urlopen(urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(),
                                                                   {"Content-Type": "application/json"}), timeout=900))
res = {}
msgs = [{"role": "user", "content": "What's the weather in Paris right now? Use the get_weather tool."}]
r = chat(msgs, tools=TOOLS, max_tokens=2000)
m = r["choices"][0]["message"]; calls = m.get("tool_calls") or []
res["tool_call"] = bool(calls)
if calls:
    msgs += [{"role": "assistant", "content": m.get("content") or "", "tool_calls": calls},
             {"role": "tool", "tool_call_id": calls[0]["id"], "content": '{"temp_c": 21, "sky": "sunny"}'}]
    r = chat(msgs, tools=TOOLS, max_tokens=2000)
    res["round_trip_21"] = "21" in (r["choices"][0]["message"].get("content") or "")
    res["round_trip_calls_again"] = bool(r["choices"][0]["message"].get("tool_calls"))
r = chat([{"role": "user", "content": "What is 23*17? Think step by step, then give the number."}], max_tokens=3000,
         chat_template_kwargs={"enable_thinking": True})
res["thinking_answer"] = "391" in (r["choices"][0]["message"].get("content") or "")
res["thinking_finish"] = r["choices"][0]["finish_reason"]
json.dump(res, open(OUT, "w")); print(json.dumps(res))
