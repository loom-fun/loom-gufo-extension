#!/usr/bin/env python3
"""Image input through gufo: color recognition, image cache identity, Responses/Messages image parts."""
import json, base64, sys, urllib.request, zlib, struct
BASE, MODEL = sys.argv[1], sys.argv[2]
def png(rgb):
    w = h = 64
    raw = b''.join(b'\x00' + bytes(rgb) * w for _ in range(h))
    c = lambda t, d: struct.pack('>I', len(d)) + t + d + struct.pack('>I', zlib.crc32(t + d) & 0xffffffff)
    return b'\x89PNG\r\n\x1a\n' + c(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0)) + c(b'IDAT', zlib.compress(raw)) + c(b'IEND', b'')
def url(rgb): return "data:image/png;base64," + base64.b64encode(png(rgb)).decode()
def post(path, body):
    try:
        return json.load(urllib.request.urlopen(urllib.request.Request(BASE + path, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600))
    except urllib.error.HTTPError as e:
        return {"error": e.read().decode()[:300]}
def ask(rgb):
    r = post("/v1/chat/completions", {"model": MODEL, "max_tokens": 64, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False},
              "messages": [{"role": "user", "content": [{"type": "text", "text": "What single color fills this image? Answer with one word."},
                                                       {"type": "image_url", "image_url": {"url": url(rgb)}}]}]})
    return (r.get("choices") or [{}])[0].get("message", {}).get("content") or r, r.get("timings", {}).get("cache_n")
red, red_cache = ask((220, 30, 30)); blue, blue_cache = ask((30, 40, 220)); red2, red2_cache = ask((220, 30, 30))
print(json.dumps({"red": red, "blue": blue, "red_again": red2, "cache_n": [red_cache, blue_cache, red2_cache]}))
ok = "red" in str(red).lower() and "blue" in str(blue).lower() and red == red2
r = post("/v1/messages", {"model": MODEL, "max_tokens": 2000, "messages": [{"role": "user", "content": [
    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64.b64encode(png((30, 200, 40))).decode()}},
    {"type": "text", "text": "What single color fills this image? One word."}]}]})
green = "".join(b.get("text", "") for b in r.get("content", []) if b.get("type") == "text") if "content" in r else r
print(json.dumps({"messages_green": green}))
print("PASS" if ok and "green" in str(green).lower() else "FAIL")
