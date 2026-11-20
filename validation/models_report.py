#!/usr/bin/env python3
"""Summarizes the declared models' validation runs into results/models.md and
results/models.json (which tools/models_table.py reads for the README).

Each run is the user flow on one machine: `gufo model install ID`,
`gufo serve llm --model ID --sessions 4`, conformance.py --logits (CONF/ID.json)
and bench.py (BENCH/ID.json); optionally loom-serve with bench.py on the same
package (LOOMSERVE/ID.json) for the Gufo / loom-serve ratio.

    validation/models_report.py CONF BENCH [LOOMSERVE] --machine TEXT [--devlab DIR --devlab-note TEXT]

--devlab DIR: conformance results (ID.json) of models run on AMD DevLab
Dispatch instead (Gufo bundle, the package fetched with loom-fetch), for
models without a local run.
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

# Single failures, explained (model id, check).
EXPLAINED = {
    ("llama-3.2-3b-instruct-loom", "tool round trip"):
        "calls the tool again instead of answering (model behavior; loom-serve does the same)",
    ("qwen3.5-2b-loom", "reasoning effort"):
        "low and high effort both reason to the length limit (the small model's reasoning loops)",
    ("ornith-1.0-35b-loom", "tool_choice named"):
        "reasoning loops when the named tool conflicts with the question (as on Loom 0.1.0, results/logits.md)",
    ("laguna-xs-2.1-loom", "shared prefix across sessions"):
        "a prefix shared by two different conversations is not reused; each conversation's own history is",
}


def load(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def main():
    def option(name):
        return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else ""
    machine, devlab, devlab_note = option("--machine"), option("--devlab"), option("--devlab-note")
    values = {machine, devlab, devlab_note}
    args = [a for a in sys.argv[1:] if not a.startswith("--") and a not in values]
    conf_dir, bench_dir = Path(args[0]), Path(args[1])
    ls_dir = Path(args[2]) if len(args) > 2 else None
    manifest = json.loads((ROOT / "extension.json").read_text())
    summary, lines = {}, []
    lines += [
        "# Declared models through Gufo",
        "",
        f"Local: each model as a user runs it, `gufo model install ID`, then `gufo serve llm --model ID --sessions 4`. {machine}",
        "",
        f"DevLab: {devlab_note}" if devlab else "",
        "",
        "Gufo checks: `conformance.py --logits` (29 checks). Throughput: `bench.py --reps 2` (greedy, thinking off;",
        "decode = one stream after a short prompt, prefill = time to first token of a ~10k-token document, 4 streams =",
        "aggregate decode). Gufo / loom-serve: the same package, preset and flags under loom-serve 0.2.0 on the same machine.",
        "",
        "| Model ID | Run | Gufo checks | Decode tok/s | Prefill tok/s | 4 streams tok/s | Gufo / loom-serve decode, prefill, 4 streams | Notes |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for model in manifest["models"]:
        mid = model["id"]
        conf, bench = load(conf_dir / f"{mid}.json"), load(bench_dir / f"{mid}.json")
        where = "local"
        if conf is None and devlab:
            conf, where = load(Path(devlab) / f"{mid}.json"), "DevLab"
        ls = load(ls_dir / f"{mid}.json") if ls_dir else None
        if conf is None:
            lines.append(f"| `{mid}` | | not run | | | | | |")
            continue
        results = conf["results"]
        passed = sum(r["status"] == "pass" for r in results)
        failed = [r for r in results if r["status"] == "fail"]
        notes = []
        for r in failed:
            why = EXPLAINED.get((mid, r["check"])) or KNOWN.get(r["check"]) or str(r.get("detail") or "")[:120]
            notes.append(f"{r['check']}: {why}")
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
        tool_free = failed and all(KNOWN.get(r["check"]) and (mid, r["check"]) not in EXPLAINED for r in failed)
        note = "no tool calls (model)" if tool_free else "; ".join(notes)
        lines.append(f"| `{mid}` | {where} | {checks} | {nums[0]} | {nums[1]} | {nums[2]} | {ratio} | {note} |")
    (HERE / "results" / "models.md").write_text("\n".join(lines) + "\n")
    (HERE / "results" / "models.json").write_text(json.dumps(summary, indent=1) + "\n")


if __name__ == "__main__":
    main()
