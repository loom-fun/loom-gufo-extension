#!/usr/bin/env python3
"""Declares the loom-fun packages in tools/models.txt as Gufo models.

For each package it pins the revision built for the Loom SDK this extension
runs (nix/loom-runtime.nix): the card's `v<sdk>` tag when it has one (loom-fun
tags the last revision for an SDK once the default branch moves to a newer
one), else the default branch. Its llm_config.json must name an SDK the
runtime supports. Then it reads its
model.json: the package files become the component that fills --model (the
model directory), and every origin the card's parameter files come from
(checkpoint, drafter, vision) becomes a component with those files at their
repository paths, which is where the extension finds them. It rewrites the
"models" array of extension.json; then run Gufo's tools/models/lock.py on this
directory to pin each file's size and SHA-256 in models.lock.json.

    python3 tools/models.py                     # needs network (Hugging Face)
    python3 /path/to/gufo/tools/models/lock.py .
    python3 tools/models.py --licenses          # refresh only the license fields
    python3 tools/models.py --check             # every pinned package can be used by the runtime
    python3 tools/models.py --add               # declare only the repositories new in models.txt

Set HF_TOKEN for higher rate limits; HF_ENDPOINT selects a mirror. LOOM_ORG=loom-fun-staging declares every
repository of models.txt from that org instead (a release candidate before an SDK promotion: validate it, never
commit it).
"""

import json
import os
from pathlib import Path
import re
import sys
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
# Package files the runtime does not read: the card's prose, reports, images.
PACKAGE_EXCLUDE = ["README.md", "REPORT.md", "info.json", ".gitattributes",
                   "*.png", "*.jpg", "*.svg"]
OPTION = {"checkpoint": "--loom-checkpoint", "drafter": "--loom-drafter",
          "vision": "--loom-vision"}
QUANT = re.compile(r"(UD-)?(I?Q[0-9]_[0-9A-Z_]+|MXFP4[0-9A-Z_]*|BF16|F16|Q8_0|Q4_0)", re.I)


def endpoint():
    return os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")


def get(url):
    request = urllib.request.Request(url)
    if os.environ.get("HF_TOKEN"):
        request.add_header("Authorization", "Bearer " + os.environ["HF_TOKEN"])
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def info(repo):
    return json.loads(get(f"{endpoint()}/api/models/{repo}"))


def raw(repo, revision, path):
    return get(f"{endpoint()}/{repo}/resolve/{revision}/{urllib.parse.quote(path)}")


def license_of(repo):
    meta = info(repo)
    data = meta.get("cardData") or {}
    value = data.get("license")
    if not value:  # a license the Hub knows only as a tag
        tags = [t.split(":", 1)[1] for t in meta.get("tags", []) if t.startswith("license:")]
        value = tags[0] if tags else None
    if value == "other" and data.get("license_name"):
        value = data["license_name"]
    return value if isinstance(value, str) else None


def runtime_sdk():
    """The hrx-loom release nix/loom-runtime.nix builds (the embed's LOOM_TEXT_EMBED_HRX_LOOM)."""
    match = re.search(r'^  version = "([^"]+)";', (ROOT / "nix" / "loom-runtime.nix").read_text(), re.M)
    if not match:
        raise SystemExit("nix/loom-runtime.nix: no version")
    return match.group(1)


def series(version):
    return ".".join(version.split(".")[:2])


def package_sdk(repo, revision):
    """The SDK release a package revision is built for (llm_config.json "sdk"; none: 0.1.0)."""
    config = json.loads(raw(repo, revision, "llm_config.json"))
    return (config.get("sdk") or {}).get("hrx_loom") or "0.1.0"


def sdk_revision(repo, sdk):
    """The package revision for `sdk`: its v<sdk> tag, else the default branch's head."""
    refs = json.loads(get(f"{endpoint()}/api/models/{repo}/refs"))
    for tag in refs.get("tags", []):
        if tag.get("name") == f"v{sdk}":
            return tag["targetCommit"]
    return info(repo)["sha"]


def check_sdk(repo, revision, sdk):
    built = package_sdk(repo, revision)
    built_series, runtime_series = series(built), series(sdk)
    # Keep this rule in step with src/loom_package.hpp.
    if built_series != runtime_series and (built_series, runtime_series) != ("0.2", "0.3"):
        raise SystemExit(f"{repo}@{revision[:8]} is built for Loom SDK {built}, the runtime is {sdk}")


def license_text(licenses):
    """The checkpoint's license first, then any other origin's that differs."""
    return "; ".join(dict.fromkeys(licenses)) or "no license stated"


def parameter_files(card):
    """[(origin, repository path)] of every parameter file the card names."""
    files = []
    for variant in card.get("variants", {}).values():
        for service in variant.get("services", {}).values():
            for parameter in (service.get("parameters") or {}).values():
                for item in parameter.get("files", []):
                    origin = item.get("origin", "")
                    if origin.startswith("@") and "/" in origin:
                        name, path = origin[1:].split("/", 1)
                        if (name, path) not in files:
                            files.append((name, path))
    return files


def model_id(card):
    name = card["model"]["id"].split("/")[-1].lower()
    return re.sub(r"[^a-z0-9._-]+", "-", name).strip("-") + "-loom"


def describe(card, files):
    name = card["model"]["id"].split("/")[-1]
    quant = None
    for origin, path in files:
        if origin == "checkpoint" and "mmproj" not in path.lower():
            match = QUANT.search(Path(path).name)
            quant = match.group(0).upper().replace("UD-", "UD-") if match else None
            break
    extras = []
    if any(origin == "drafter" or Path(path).name.lower().startswith(("mtp-", "dspark-")) for origin, path in files):
        extras.append("speculative decoding")
    if any(origin == "vision" or "mmproj" in path.lower() for origin, path in files):
        extras.append("vision")
    text = f"{name} on AMD Loom ({quant + ' ' if quant else ''}GGUF"
    return text + (", " + ", ".join(extras) if extras else "") + ")"


def declare(repo, sdk):
    revision = sdk_revision(repo, sdk)
    check_sdk(repo, revision, sdk)
    card = json.loads(raw(repo, revision, "model.json"))
    files = parameter_files(card)
    origins = card.get("origins", {})
    licenses = []
    components = [{
        "role": "package",
        "option": "--model",
        "install": "required",
        "source": {"hf": repo, "revision": revision},
        "exclude": PACKAGE_EXCLUDE,
        "entry": ".",
    }]
    for origin in dict.fromkeys(name for name, _ in files):
        source = origins.get(origin, {})
        if source.get("kind") != "hf" or origin not in OPTION:
            raise SystemExit(f"{repo}: origin {origin} is not supported")
        paths = [path for name, path in files if name == origin]
        component = {
            "role": origin,
            "option": OPTION[origin],
            "install": "required",
            "source": {"hf": source["repository"], "revision": source["revision"]},
            "include": paths,
        }
        if len(paths) > 1:
            component["entry"] = "."
        components.append(component)
        value = license_of(source["repository"])
        if origin == "checkpoint":
            licenses.insert(0, value or "no license stated")
        elif value != (licenses[0] if licenses else None):
            licenses.append(f"{origin} ({source['repository']}): {value or 'no license stated'}")
    quant = next((QUANT.search(Path(path).name) for origin, path in files
                  if origin == "checkpoint" and "mmproj" not in path.lower()), None)
    return {
        "id": model_id(card),
        "_quant": quant.group(0).lower() if quant else "",
        "provider": "loom-text",
        "license": license_text(licenses),
        "description": describe(card, files),
        "components": components,
    }


def relicense():
    """Rewrites only the license fields (keeps every pinned revision)."""
    path = ROOT / "extension.json"
    manifest = json.loads(path.read_text())
    for model in manifest["models"]:
        licenses = []
        for component in model["components"][1:]:
            value = license_of(component["source"]["hf"])
            if component["role"] == "checkpoint":
                licenses.insert(0, value or "no license stated")
            elif value != (licenses[0] if licenses else None):
                licenses.append(f"{component['role']} ({component['source']['hf']}): {value or 'no license stated'}")
        model["license"] = license_text(licenses)
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def check():
    """Every pinned package is built for the runtime's SDK release."""
    sdk = runtime_sdk()
    manifest = json.loads((ROOT / "extension.json").read_text())
    for model in manifest["models"]:
        source = model["components"][0]["source"]
        check_sdk(source["hf"], source["revision"], sdk)
    print(f"{len(manifest['models'])} packages compatible with Loom SDK {sdk}", file=sys.stderr)


def main():
    if "--licenses" in sys.argv:
        return relicense()
    if "--check" in sys.argv:
        return check()
    sdk = runtime_sdk()
    # One repository per line, optionally followed by the Gufo model ID to give it.
    lines = [line.split() for line in (ROOT / "tools" / "models.txt").read_text().splitlines()
             if line.strip() and not line.startswith("#")]
    org = os.environ.get("LOOM_ORG")
    if org:
        lines = [[f"{org}/{line[0].split('/', 1)[1]}", *line[1:]] for line in lines]
    repos = [line[0] for line in lines]
    names = {line[0]: line[1] for line in lines if len(line) > 1}
    path = ROOT / "extension.json"
    manifest = json.loads(path.read_text())
    # --add: declare only the repositories not yet in extension.json; every existing pin stays.
    models = manifest.get("models", []) if "--add" in sys.argv else []
    known = {model["components"][0]["source"]["hf"] for model in models}
    for repo in repos:
        if repo in known:
            continue
        model = declare(repo, sdk)
        quant = model.pop("_quant")
        if repo in names:
            model["id"] = names[repo]
        if any(other["id"] == model["id"] for other in models) and quant:
            # A second package of the same model: its quantization tells them apart.
            model["id"] = model["id"].removesuffix("-loom") + f"-{quant}-loom"
        if any(other["id"] == model["id"] for other in models):
            raise SystemExit(f"{repo}: model id {model['id']} is taken")
        models.append(model)
        print(f"{model['id']:40} {repo}", file=sys.stderr)
    manifest["models"] = models
    path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
