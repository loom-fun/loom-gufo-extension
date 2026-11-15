#!/usr/bin/env python3
"""gufo+Loom vs loom-serve table from results/all2: bench_table.py [DIR]"""
import json, glob, os, sys, statistics
D = sys.argv[1] if len(sys.argv) > 1 else "results/all2"
order = [l.strip() for l in open("fetch-list-sorted.txt") if l.strip()]
rows, ratios = [], {"decode": [], "prefill": [], "c4": []}
for name in order:
    lf, gf = f"{D}/loom-serve-{name}.json", f"{D}/gufo-{name}.json"
    if not (os.path.exists(lf) and os.path.exists(gf)):
        rows.append(f"| {name} | missing | | | | |"); continue
    L, G = json.load(open(lf)), json.load(open(gf))
    same = sum(a == b for a, b in zip(L["correctness"], G["correctness"]))
    cells = []
    for k in ("decode", "prefill", "c4"):
        l, g = L[k]["tok_s"], G[k]["tok_s"]
        ratios[k].append(g / l); cells.append(f"{l:.1f} / {g:.1f} ({100 * g / l:.0f}%)")
    ttft = f"{L['decode']['ttft_s'] * 1000:.0f} / {G['decode']['ttft_s'] * 1000:.0f}"
    rows.append(f"| {L['model']} | {same}/5 | " + " | ".join(cells) + f" | {ttft} |")
print("| Model | Greedy identical | Decode tok/s (loom-serve / gufo) | Prefill tok/s, ~2.1k prompt | 4-stream tok/s | TTFT ms |")
print("|---|---|---|---|---|---|")
print("\n".join(rows))
for k, v in ratios.items():
    if v: print(f"\n{k}: median {100 * statistics.median(v):.0f}%, min {100 * min(v):.0f}%, max {100 * max(v):.0f}% over {len(v)} models", end="")
print()
