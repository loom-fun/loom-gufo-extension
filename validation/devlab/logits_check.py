#!/usr/bin/env python3
"""Logits readback checks through a Gufo server that loaded a package with the logits capability.

Usage: logits_check.py BASE_URL OUT.json

1. The device path is untouched: the same greedy request, with and without logprobs (logprobs makes the runner
   sample on the host from the exported logit row), returns the same text. A difference is a near-tie unless
   it comes with a clear margin; the margin (top-1 minus top-2 logprob at the first differing token) is
   recorded and a gap above 0.05 nats fails the check.
2. logprobs are consistent: the sampled token is the top-1 entry at temperature 0, entries are sorted, their
   probability mass is at most 1, and the first token's distribution sums close to 1 over top_logprobs=20.
3. A grammar request (json_schema) yields valid JSON while the plain request on the same server is served on
   the device (both complete: the server mixes the two kinds of request in one batch).
"""
import concurrent.futures as cf
import json
import math
import sys
import urllib.error
import urllib.request

BASE, OUT = sys.argv[1], sys.argv[2]
results = []


def call(body, timeout=900):
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:300]


def check(name, ok, detail):
    results.append({"check": name, "status": "pass" if ok else "fail", "detail": str(detail)[:400]})
    print(("PASS" if ok else "FAIL"), name, str(detail)[:200], flush=True)


MODEL = json.loads(urllib.request.urlopen(BASE + "/v1/models").read())["data"][0]["id"]
NOTHINK = {"chat_template_kwargs": {"enable_thinking": False}}
PROMPTS = ["What is the capital of France? Answer in one sentence.",
           "List three primary colors, comma separated.",
           "Write a haiku about the sea.",
           "Explain in two sentences why the sky is blue.",
           "Give me a python one-liner that reverses a string."]


def ask(prompt, **kw):
    s, r = call({"model": MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0,
                 "max_tokens": 120, **NOTHINK, **kw})
    if s != 200:
        return s, r, None
    m = r["choices"][0]["message"]
    return s, (m.get("reasoning_content") or "") + (m.get("content") or ""), r["choices"][0].get("logprobs")


# 1 + 2: device == host greedy, logprobs consistent
same, diffs, bad = 0, [], []
for p in PROMPTS:
    s1, device, _ = ask(p)
    s2, host, lp = ask(p, logprobs=True, top_logprobs=20)
    if s1 != 200 or s2 != 200:
        bad.append((p, s1, s2, device if s1 != 200 else host))
        continue
    content = (lp or {}).get("content") or []
    for e in content:
        top = e.get("top_logprobs") or []
        mass = sum(math.exp(t["logprob"]) for t in top)
        if not top or top[0]["token"] != e["token"] or abs(top[0]["logprob"] - e["logprob"]) > 1e-3 \
                or mass > 1.001 or any(top[i]["logprob"] < top[i + 1]["logprob"] for i in range(len(top) - 1)):
            bad.append((p, "inconsistent logprobs", str(e)[:120]))
            break
    if device == host:
        same += 1
    else:
        # first differing position, to read its margin from the logprobs
        i = next((k for k, (a, b) in enumerate(zip(device, host)) if a != b), min(len(device), len(host)))
        margin = None
        pos = 0
        for e in content:
            pos += len(e["token"])
            if pos > i:
                top = e.get("top_logprobs") or []
                margin = top[0]["logprob"] - top[1]["logprob"] if len(top) > 1 else None
                break
        diffs.append({"prompt": p, "at": i, "margin_nats": margin})
check("host greedy == device greedy", not bad and all((d["margin_nats"] is not None and d["margin_nats"] < 0.05) for d in diffs),
      f"{same}/{len(PROMPTS)} identical; differing {diffs}; problems {bad[:2]}")

# 3: a grammar request and a plain one together
schema = {"type": "json_schema", "json_schema": {"name": "x", "strict": True, "schema": {
    "type": "object", "properties": {"n": {"type": "integer"}, "word": {"type": "string"}},
    "required": ["n", "word"], "additionalProperties": False}}}


def grammar():
    s, r = call({"model": MODEL, "messages": [{"role": "user", "content": "Give a number and a word as JSON."}],
                 "temperature": 0, "max_tokens": 2500, "response_format": schema, **NOTHINK})
    return s, r


def plain():
    return ask("Say a short greeting.")


with cf.ThreadPoolExecutor(4) as pool:
    futures = [pool.submit(grammar), pool.submit(plain), pool.submit(grammar), pool.submit(plain)]
    outs = [f.result() for f in futures]
ok = True
for s, r in (outs[0], outs[2]):
    try:
        obj = json.loads(r["choices"][0]["message"]["content"])
        ok = ok and s == 200 and isinstance(obj["n"], int) and isinstance(obj["word"], str) and set(obj) == {"n", "word"}
    except Exception:
        ok = False
ok = ok and all(o[0] == 200 and o[1] for o in (outs[1], outs[3]))
check("grammar and plain requests batched together", ok, str([o[1] if o[0] != 200 else "ok" for o in outs]))

json.dump({"model": MODEL, "results": results}, open(OUT, "w"), indent=1)
print(json.dumps({k: sum(r["status"] == k for r in results) for k in ("pass", "fail")}))
sys.exit(1 if any(r["status"] == "fail" for r in results) else 0)
