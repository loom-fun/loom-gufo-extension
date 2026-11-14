#!/usr/bin/env python3
"""Disk tier across a restart: disk_check.py BASE OUT.json first|second (second compares and writes the verdict)."""
import json, subprocess, sys
BASE, OUT, phase = sys.argv[1:4]
run = lambda: json.loads(subprocess.check_output([sys.executable, __import__("os").path.join(__import__("os").path.dirname(__file__), "disk_test.py"), BASE]))
if phase == "first":
    a, b = run(), run()
    json.dump({"first": a, "first_again": b}, open(OUT, "w"))
else:
    d = json.load(open(OUT)); c = run()
    d["after_restart"] = c
    d["status"] = "pass" if c["reuse"] == "disk" and c["cache_n"] > 0 and c["content"] == d["first"]["content"] else "fail"
    json.dump(d, open(OUT, "w")); print(d["status"], c)
