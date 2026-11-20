#!/usr/bin/env python3
"""Summarizes the declared models' validation runs into results/models.md and
results/models.json (which tools/models_table.py reads for the README).

Each run is the user flow on one machine: `gufo model install ID`,
`gufo serve llm --model ID --sessions 4`, conformance.py --logits (CONF/ID.json)
and bench.py (BENCH/ID.json); optionally loom-serve with bench.py on the same
package (LOOMSERVE/ID.json) for the Gufo / loom-serve ratio.

    validation/models_report.py CONF BENCH [LOOMSERVE] --machine TEXT
"""

import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
# Failures that are the model's own behavior (loom-serve shows the same).
KNOWN = {
    "tool call (auto)": "the package declares no tool-call format",
    "streamed tool call": "the package declares no tool-call format",
    "tool_choice required": "the package declares no tool-call format",
    "tool_choice named": "the package declares no tool-call format",
    "strict tool schema": "the package declares no tool-call format",
    "Responses API (text + function call)": "the package declares no tool-call format",
    "Anthropic Messages (text + tool_use)": "the package declares no tool-call format",
}


def load(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    machine = sys.argv[sys.argv.index("--machine") + 1] if "--machine" in sys.argv else ""
    args = [a for a in args if a != machine]
    conf_dir, bench_dir = Path(args[0]), Path(args[1])
    ls_dir = Path(args[2]) if len(args) > 2 else None
    manifest = json.loads((ROOT / "extension.json").read_text())
    summary, lines = {}, []
    lines += [
        "# Declared models through Gufo",
        "",
        f"Each model as a user runs it: `gufo model install ID`, then `gufo serve llm --model ID --sessions 4`. {machine}",
        "",
        "Gufo checks: `conformance.py --logits` (29 checks). Throughput: `bench.py --reps 2` (greedy, thinking off;",
        "decode = one stream after a short prompt, prefill = time to first token of a ~10k-token document, 4 streams =",
        "aggregate decode). Gufo / loom-serve: the same package, preset and flags under loom-serve 0.2.0 on the same machine.",
        "",
        "| Model ID | Gufo checks | Decode tok/s | Prefill tok/s | 4 streams tok/s | Gufo / loom-serve decode, prefill, 4 streams | Notes |",
        "|---|---|---|---|---|---|---|",
    ]
    for model in manifest["models"]:
        mid = model["id"]
        conf, bench = load(conf_dir / f"{mid}.json"), load(bench_dir / f"{mid}.json")
        ls = load(ls_dir / f"{mid}.json") if ls_dir else None
        if conf is None:
            lines.append(f"| `{mid}` | not run | | | | | |")
            continue
        results = conf["results"]
        passed = sum(r["status"] == "pass" for r in results)
        failed = [r for r in results if r["status"] == "fail"]
        notes = []
        for r in failed:
            notes.append(f"{r['check']}: {KNOWN.get(r['check'], str(r.get('detail') or '')[:120])}")
        checks = f"{passed}/{len(results)}"
        summary[mid] = {"conformance": checks, "failed": [r["check"] for r in failed]}
        nums = ["", "", ""]
        ratio = ""
        if bench:
            nums = [f"{bench['decode']['tok_s']:,.1f}", f"{bench['prefill']['tok_s']:,.0f}", f"{bench['c4']['tok_s']:,.1f}"]
            summary[mid]["bench"] = {k: bench[k]["tok_s"] for k in ("decode", "prefill", "c4")}
            if ls:
                parts = [bench[k]["tok_s"] / ls[k]["tok_s"] for k in ("decode", "prefill", "c4")]
                ratio = ", ".join(f"{p * 100:.0f} %" for p in parts)
                summary[mid]["vs_loom_serve"] = dict(zip(("decode", "prefill", "c4"), parts))
        tool_free = failed and all(KNOWN.get(r["check"]) for r in failed)
        note = "no tool calls (model)" if tool_free else "; ".join(notes)
        lines.append(f"| `{mid}` | {checks} | {nums[0]} | {nums[1]} | {nums[2]} | {ratio} | {note} |")
    (HERE / "results" / "models.md").write_text("\n".join(lines) + "\n")
    (HERE / "results" / "models.json").write_text(json.dumps(summary, indent=1) + "\n")


if __name__ == "__main__":
    main()
