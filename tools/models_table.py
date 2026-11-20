#!/usr/bin/env python3
"""Writes the README's model table from extension.json, models.lock.json and
each package card's info.json (at the pinned revision), plus the Gufo
validation results when they exist (validation/results/models.json).

    python3 tools/models_table.py            # prints the Markdown table
    python3 tools/models_table.py --readme   # replaces it in README.md

Needs network (Hugging Face) for info.json.
"""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import models as declarations  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BEGIN, END = "<!-- models table: tools/models_table.py -->", "<!-- end models table -->"


def size(model, lock):
    total = sum(f["size"] for component in lock[model["id"]].values() for f in component["files"])
    return f"{total / 1e9:.1f} GB" if total >= 1e9 else f"{total / 1e6:.0f} MB"


def features(card):
    flags = card.get("card", {}).get("features", {})
    names = [("tool_calling", "tools"), ("thinking", "reasoning"),
             ("image_input", "vision"), ("speculative_decoding", "drafting")]
    return ", ".join(label for key, label in names
                     if (flags.get(key) or {}).get("status") == "yes")


def metric(card, name):
    """(Loom, llama.cpp ROCm) tokens/s on Strix Halo from the card's session."""
    performance = card.get("devices", {}).get("strix-halo", {}).get("performance", {})
    for session in reversed(performance.get("sessions") or []):
        value = session.get("metrics", {}).get(name) or {}
        if value.get("loom") is not None:
            return value["loom"], value.get("baseline")
    return None


def cell(pair):
    if pair is None:
        return "—"
    loom, base = pair
    if not base:
        return f"{loom:,.0f} (no llama.cpp run)"
    return f"{loom:,.0f} vs {base:,.0f} ({(loom / base - 1) * 100:+.0f} %)"


def table():
    manifest = json.loads((ROOT / "extension.json").read_text())
    lock = json.loads((ROOT / "models.lock.json").read_text())["models"]
    results_path = ROOT / "validation" / "results" / "models.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    rows = [
        "| Gufo model ID | Model | Download | Features | Decode tok/s, Loom vs llama.cpp | "
        "Prefill tok/s, 32k prompt | Gufo checks |",
        "|---|---|---|---|---|---|---|",
    ]
    for model in manifest["models"]:
        package = model["components"][0]["source"]
        card = json.loads(declarations.raw(package["hf"], package["revision"], "info.json"))
        name = json.loads(declarations.raw(package["hf"], package["revision"], "model.json"))["model"]["id"]
        checks = results.get(model["id"], {}).get("conformance")
        rows.append(
            f"| `{model['id']}` | [{name}](https://huggingface.co/{package['hf']}) | {size(model, lock)} | "
            f"{features(card)} | {cell(metric(card, 'decode'))} | {cell(metric(card, 'prefill_32k'))} | "
            f"{checks or '—'} |")
    return "\n".join(rows)


def main():
    text = table()
    if "--readme" not in sys.argv:
        print(text)
        return
    readme = ROOT / "README.md"
    current = readme.read_text()
    start, end = current.index(BEGIN) + len(BEGIN), current.index(END)
    readme.write_text(current[:start] + "\n" + text + "\n" + current[end:])


if __name__ == "__main__":
    main()
