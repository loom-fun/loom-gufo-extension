# loom-gufo-extension

48 open models, from Qwen3 0.6B to Qwen3.6 35B-A3B, gemma 4, gpt-oss 20B, Llama 3.2, LFM, Nemotron and more, served by
[Gufo](https://github.com/gufo-org/gufo) on [AMD Loom](https://github.com/ROCm/hrx-system) kernels. Each model is a
[loom-fun](https://huggingface.co/loom-fun) package of Loom programs for an unmodified GGUF; Gufo keeps everything
else: its OpenAI, Anthropic and llama-server APIs, scheduler, prompt cache, tool calls, reasoning, grammars and
`gufo model`. On Strix Halo, Loom decodes faster than llama.cpp ROCm on 43 of the 47 measured packages (the other 4
tie, within ±3 %) and prefills a 32k-token prompt faster on 45 of 46, by a median of 17 % and 53 % (see
[Models](#models)).

Maintainer: [@jeremyfowers](https://github.com/jeremyfowers). Platform: Linux x86-64 with the amdgpu driver, tested
on Strix Halo (gfx1151). Packages that publish Radeon AI PRO R9700 or 16 GB gfx1201 presets get them automatically on
those GPUs, where Loom serves them, but this release's Gufo runs are Strix Halo only.
License: MIT (`LICENSE`). Needs Gufo with extension API 0.3.35 (gufo-org/gufo main from
[ec6d3d07](https://github.com/gufo-org/gufo/commit/ec6d3d07)).

## Quick start

With [Nix](https://nixos.org/download/) (flakes on, and Gufo's binary cache: see [Install](#install)):

```sh
nix profile install github:loom-fun/loom-gufo-extension    # gufo with this extension (or `nix run ...#default --`)
gufo model install qwen3-4b-loom                            # the package and its GGUF, verified (3.1 GB)
gufo serve llm --model qwen3-4b-loom                        # 127.0.0.1:8080
```

```sh
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-4B","messages":[{"role":"user","content":"Hello!"}]}'
```

`gufo model list` shows every model; pick any ID from the table below. The extension serves the preset the package
publishes for your GPU (`loom: preset chat for gfx1151` in the log; `--loom-preset` picks another).

## Install

Gufo's binary cache avoids building ROCm's LLVM from source (hours). Put these lines in `/etc/nix/nix.conf` (and
restart the Nix daemon), or in `NIX_CONFIG`:

```
experimental-features = nix-command flakes
extra-substituters = https://gufo.cachix.org
extra-trusted-public-keys = gufo.cachix.org-1:rfZe3n4lHINgyoACKzs5EthIZmBNvEyCCtufUpth0PE=
```

The extension links `libloom_text_embed`, Loom's text model layer behind a small C API (`loom-embed/`), built from the
pinned hrx-system release by `nix/runtime.nix` with the nixpkgs Gufo pins (a few minutes; it needs no ROCm). Three
ways to get Gufo with it:

- **This flake** (Quick start): Gufo at the revision in `flake.lock`, with the extension built as a module and the
  library. `nix build github:loom-fun/loom-gufo-extension#runtime` builds the library alone.
- **Your Gufo, as a module** from the [Gufo extension registry](https://github.com/gufo-org/gufo-extension), with the
  library on `CMAKE_PREFIX_PATH` for the build:

  ```sh
  nix profile install github:gufo-org/gufo#sdk
  LOOM=$(nix build --no-link --print-out-paths github:loom-fun/loom-gufo-extension#runtime)
  CMAKE_PREFIX_PATH=$LOOM gufo extension install loom
  gufo diagnose --section extensions              # lists "provider loom-text: available"
  ```

  `gufo extension install https://github.com/loom-fun/loom-gufo-extension@<commit or tag>` installs a revision that
  is not in the registry. Keep `CMAKE_PREFIX_PATH=$LOOM` for `gufo extension update` after upgrading Gufo; serving
  needs nothing else (the module finds the library through its run path). The library is built with the nixpkgs of
  the Gufo in this repository's `flake.lock`; to match another Gufo exactly, build it with that one:
  `nix build --impure --expr 'import ./nix/runtime.nix { gufo = builtins.getFlake "github:gufo-org/gufo/<rev>"; }'`.
- **Compiled into your own Gufo build:**

  ```sh
  git clone https://github.com/gufo-org/gufo
  git clone https://github.com/loom-fun/loom-gufo-extension
  cd gufo
  LOOM=$(nix build --impure --no-link --print-out-paths --expr \
    "import $PWD/../loom-gufo-extension/nix/runtime.nix { gufo = builtins.getFlake \"path:$PWD\"; }")
  nix develop --impure --expr "(builtins.getFlake \"path:$PWD\").packages.x86_64-linux.default" --command bash -c "
    export PKG_CONFIG_PATH=$LOOM/lib/pkgconfig:\$PKG_CONFIG_PATH && eval \"\$preConfigure\" &&
    cmake -S . -B build -G Ninja \$cmakeFlags -DENGINE_ENABLE_HIP=ON -DGUFO_EXTENSIONS=loom \
      -DGUFO_EXTENSION_DIRS=$PWD/../loom-gufo-extension &&
    cmake --build build --target gufo"
  ```

No Nix on the machine (or no root)? Run the same commands in the `nixos/nix` image, keeping `/nix` in a named volume so
Gufo and the library are still there when you serve:

```sh
docker run --rm -it -v gufo-nix:/nix -v $PWD:$PWD -w $PWD -e NIX_CONFIG="$(printf '%s\n' \
  'experimental-features = nix-command flakes' 'extra-substituters = https://gufo.cachix.org' \
  'extra-trusted-public-keys = gufo.cachix.org-1:rfZe3n4lHINgyoACKzs5EthIZmBNvEyCCtufUpth0PE=')" nixos/nix bash
```

To serve from it, add `--device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined --ulimit memlock=-1
--network host`, and keep the model directory (`GUFO_MODELS_DIR`) on a volume.

## Models

`gufo model install <ID>` downloads the loom-fun package and the GGUF files its card names (the checkpoint, and a
drafter or vision projector where the package uses one), pinned by revision, size and SHA-256 in `models.lock.json`,
and reuses files already in the Hugging Face cache. The extension finds those files in the model directory, so
`--model <ID>` is all `gufo serve`, `prompt`, `chat` and `bench` need. A package fetched with `loom-fetch
huggingface:loom-fun/<repo>:<preset> --output=<dir>` (`pipx install 'hrx-loom[hf]==0.2.0'`) works too: `--model <dir>`.

The extension bundles no weights. They keep their upstream license (Apache 2.0 for Qwen, the Gemma terms, the Llama 3.2
Community License, LFM 1.0, ...), which `gufo model list --json` and each package card name. Use a model only under its
license.

Throughput: the package cards' measurements on AMD DevLab Strix Halo runners (Loom 0.2.0 and llama.cpp ROCm b1374, same
GGUF, same runner type, out of the box; the cards link each run and add Vulkan, longer contexts and the R9700). Through
Gufo, Loom decoded at 99-105 % of loom-serve's speed on the same machine (Loom 0.1.0, `DESIGN.md`); Gufo's own
throughput per model is in `validation/results/models.md`. Gufo checks: Gufo's 29 API
conformance checks with `--logits`, on the installed model (`validation/results/models.md` explains each failure).

<!-- models table: tools/models_table.py -->
<!-- end models table -->

**Logits:** forced or named `tool_choice`, strict tool schemas, `response_format`, `logprobs`, `min_p` and penalties
sample on the host from the package's logits (every package above returns them; the log says `loom: logits on`).
Those requests run without drafting, so they decode somewhat slower; every other request keeps the device path.

## How it works

Gufo serves a Loom package like any other model: its server, scheduler, prompt cache, parsers and API endpoints are
unchanged. The extension runs the package's own kernels, chat template, tokenizer and drafter through a small C
library (`loom-embed/`) over Loom's text model layer. Details: `DESIGN.md`.

Options: `--loom-device URI`, `--loom-memory-bytes N` (default: none, and Loom's memory pool is committed at load, as
the loom-fun cards serve; N makes it an elastic pool with that budget, which prefills slower on 0.2.0 and can refuse
long prompts when small), `--loom-preset NAME` (only for a package directory that holds several presets),
`--loom-logits auto|on|off`. `--cache-disk DIR` keeps the prompt cache across restarts (its checkpoint images need the
elastic pool, which the extension then uses).

## Limits

- In a container, run with `--security-opt seccomp=unconfined` (Loom's GPU memory reservation needs it).
- Loom streams weights from disk through pinned buffers, which need a locked-memory limit (`ulimit -l`) of at least
  64 MiB. Below that (stock Ubuntu: 8 MiB, also Docker's default) the extension reads the weights into host memory at
  load instead, as `loom-serve --parameter_mode=preload` does, so loading takes about the weights' size in extra RAM.
  Raise the limit (`ulimit -l unlimited`, `LimitMEMLOCK=infinity`, Docker `--ulimit memlock=-1`) to stream them.
- gpt-oss and Glimmer: automatic tool choice is not grammar-constrained (the package parses the calls); required and named are.

## Quality

`validation/results/` holds the evidence: Gufo's 29 conformance checks per package (`conformance.md`), throughput
against llama.cpp (`bench.md`) and the logits features (`logits.md`). Known gaps, all model behavior that loom-serve
shows too:

- gemma-3-1b, HyperCLOVAX-1.5B and Qwen2.5-VL-7B never call tools (their packages declare no tool-call format), so
  the 7 tool checks fail on them.
- Llama-3.2-3B calls the tool again instead of answering; Qwen3.5-0.8B's reasoning can loop to the length limit.
- Requests that need logits decode somewhat slower (host sampling, no drafting).

## Releases

| Version | Gufo | Notes |
|---|---|---|
| 0.3.0 | extension API >= 0.3.35 (gufo-org/gufo main from ec6d3d07) | 48 loom-fun packages as Gufo models (`gufo model install`); the package's preset for the GPU chosen automatically; packages served as published, with `--loom-checkpoint`/`--loom-drafter`/`--loom-vision`; `flake.nix` (Gufo with this extension, and the Loom library) |
| 0.2.0 | extension API >= 0.3.35 (gufo-org/gufo main from ec6d3d07) | Loom SDK 0.2.0 (hrx-system v0.2.0) on Gufo main; pinned hrx-system downloads as Nix `fetchurl`s |
| 0.1.0 | gufo-org/gufo#1243 branch | Loom SDK 0.1.0 prototype |

## Development

- Tests (no GPU, against a fake `libloom_text_embed`): `GUFO=<gufo checkout> tests/build_test.sh` runs the runner's
  host-sampling tests and `tests/package_test.cpp` (presets, parameter files by origin). Needs g++ 13+ and Cargo.
- Models: add a loom-fun repository to `tools/models.txt`, then `python3 tools/models.py` (pins each package's revision
  and its origins into `extension.json`), `python3 <gufo>/tools/models/lock.py .` (`models.lock.json`) and
  `python3 tools/models_table.py --readme` (the table above). Validate each with `validation/conformance.py --logits` on
  `gufo serve llm --model <ID>`; `validation/models_report.py` writes `validation/results/models.md`.
- hrx-system pin: after changing it in `nix/loom-runtime.nix`, regenerate its downloads with
  `python3 nix/hrx-deps.py generate <hrx-system checkout> nix/hrx-deps.nix`.
- `nix flake lock --update-input gufo` moves the flake to a newer Gufo.
