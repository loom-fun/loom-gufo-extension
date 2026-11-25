# loom-gufo-extension

52 open models, from Qwen3 0.6B to Qwen3.6 35B-A3B, gemma 4, gpt-oss 20B, Llama 3.2, LFM, Nemotron and more, served by
[Gufo](https://github.com/gufo-org/gufo) on [AMD Loom](https://github.com/ROCm/hrx-system) kernels. Each model is a
[loom-fun](https://huggingface.co/loom-fun) package of Loom programs for an unmodified GGUF; Gufo keeps everything
else: its OpenAI, Anthropic and llama-server APIs, scheduler, prompt cache, tool calls, reasoning, grammars and
`gufo model`. On Strix Halo, Loom decodes faster than llama.cpp ROCm on 46 of the 51 measured packages (the other 5
tie, within ±3 %) and prefills a 32k-token prompt faster on 48 of 49, by a median of 18 % and 54 % (see
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
huggingface:loom-fun/<repo>:<preset> --output=<dir>` (`pipx install 'hrx-loom[hf]==0.3.0'`) works too: `--model <dir>`.

The extension bundles no weights. They keep their upstream license (Apache 2.0 for Qwen, the Gemma terms, the Llama 3.2
Community License, LFM 1.0, ...), which each model's `license` (`gufo model list --json`) names, with a drafter's
when it differs. The loom-fun packages themselves (their Loom programs and cards, generated by
[loom-skills](https://github.com/loom-fun/loom-skills)) are MIT. Use a model only under its weights' license.

Throughput: the package cards' measurements on AMD DevLab Strix Halo runners (Loom 0.2.0 and llama.cpp ROCm b1374, same
GGUF, same runner type, out of the box; the cards link each run and add Vulkan, longer contexts and the R9700). Through
Gufo, Loom decoded at 99-105 % of loom-serve's speed on the same machine (Loom 0.1.0, `DESIGN.md`); Gufo's own
throughput per model is in `validation/results/models.md`. Gufo checks: Gufo's 29 API
conformance checks with `--logits`, on the installed model (`validation/results/models.md` explains each failure).

<!-- models table: tools/models_table.py -->
| Gufo model ID | Model | Download | Features | Decode tok/s, Loom vs llama.cpp | Prefill tok/s, 32k prompt | Gufo checks |
|---|---|---|---|---|---|---|
| `qwen3-0.6b-loom` | [Qwen/Qwen3-0.6B](https://huggingface.co/loom-fun/unsloth-Qwen3-0.6B-loom) | 406 MB | tools, reasoning | 308 vs 266 (+16 %) | 3,981 vs 2,717 (+47 %) | 29/29 |
| `qwen2.5-0.5b-instruct-loom` | [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/loom-fun/bartowski-Qwen2.5-0.5B-Instruct-loom) | 406 MB | tools | 320 vs 293 (+9 %) | 7,949 vs 6,501 (+22 %) | 29/29 |
| `qwen3.5-0.8b-loom` | [Qwen/Qwen3.5-0.8B](https://huggingface.co/loom-fun/unsloth-Qwen3.5-0.8B-loom) | 751 MB | tools, reasoning, vision | 265 vs 202 (+31 %) | 8,870 vs 5,639 (+57 %) | 29/29 |
| `gemma-3-1b-it-loom` | [google/gemma-3-1b-it](https://huggingface.co/loom-fun/ggml-org-gemma-3-1b-it-loom) | 825 MB |  | 194 vs 168 (+16 %) | — | 21/28 |
| `qwen2.5-1.5b-instruct-loom` | [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/loom-fun/bartowski-Qwen2.5-1.5B-Instruct-loom) | 995 MB | tools | 166 vs 156 (+6 %) | 3,413 vs 2,576 (+32 %) | 29/29 |
| `lfm3-omni-2b-loom` | [LiquidAI/LFM3-Omni-2B](https://huggingface.co/loom-fun/ggml-org-LFM3-Omni-2B-loom) | 1.1 GB | tools, reasoning | 170 vs 148 (+15 %) | 5,744 vs 4,330 (+33 %) | 29/29 |
| `lfm3-1.7b-loom` | [LiquidAI/LFM3-1.7B](https://huggingface.co/loom-fun/bartowski-LFM3-1.7B-loom) | 1.1 GB | tools, reasoning | 162 vs 137 (+18 %) | 5,635 vs 4,128 (+36 %) | 29/29 |
| `hyperclovax-seed-text-instruct-1.5b-loom` | [naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B](https://huggingface.co/loom-fun/rippertnt-HyperCLOVAX-SEED-Text-Instruct-1.5B-loom) | 1.1 GB |  | 169 vs 156 (+8 %) | 3,468 vs 2,308 (+50 %) | 21/28 |
| `qwen3-1.7b-loom` | [Qwen/Qwen3-1.7B](https://huggingface.co/loom-fun/ggml-org-Qwen3-1.7B-loom) | 1.3 GB | tools, reasoning | 148 vs 144 (+2 %) | 3,113 vs 2,030 (+53 %) | 29/29 |
| `qwen3.5-2b-loom` | [Qwen/Qwen3.5-2B](https://huggingface.co/loom-fun/unsloth-Qwen3.5-2B-loom) | 2.0 GB | tools, reasoning, vision | 134 vs 118 (+14 %) | 5,359 vs 3,489 (+54 %) | 28/29 |
| `llama-3.2-3b-instruct-loom` | [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/loom-fun/unsloth-Llama-3.2-3B-Instruct-loom) | 2.0 GB | tools | 97 vs 90 (+8 %) | 1,810 vs 1,075 (+68 %) | 28/29 |
| `qwen3-4b-instruct-2507-loom` | [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/loom-fun/unsloth-Qwen3-4B-Instruct-2507-loom) | 2.5 GB | tools, drafting | 78 vs 74 (+6 %) | 1,269 vs 867 (+46 %) | 29/29 |
| `spark-x2.5-4b-loom` | [XHToken/Spark-X2.5-4B](https://huggingface.co/loom-fun/XHToken-Spark-X2.5-4B-loom) | 2.6 GB | tools, reasoning | 72 vs 61 (+18 %) | 2,174 vs 1,274 (+71 %) | 29/29 |
| `spark-x2.6-4b-loom` | [XHToken/Spark-X2.6-4B](https://huggingface.co/loom-fun/bartowski-Spark-X2.6-4B-loom) | 2.6 GB | tools, reasoning | 70 vs 62 (+13 %) | 2,130 vs 1,340 (+59 %) | 29/29 |
| `qwen3-4b-loom` | [Qwen/Qwen3-4B](https://huggingface.co/loom-fun/ggml-org-Qwen3-4B-loom) | 3.1 GB | tools, reasoning, drafting | 100 vs 74 (+36 %) | 1,263 vs 868 (+45 %) | 29/29 |
| `qwen3-vl-4b-instruct-loom` | [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/loom-fun/unsloth-Qwen3-VL-4B-Instruct-loom) | 3.3 GB | tools, vision | 78 vs 71 (+9 %) | 1,265 vs 838 (+51 %) | 29/29 |
| `qwen3.5-4b-loom` | [Qwen/Qwen3.5-4B](https://huggingface.co/loom-fun/unsloth-Qwen3.5-4B-loom) | 4.1 GB | tools, reasoning, vision, drafting | 88 vs 56 (+57 %) | 2,322 vs 1,353 (+72 %) | 29/29 |
| `gemma-4-e2b-it-loom` | [google/gemma-4-E2B-it](https://huggingface.co/loom-fun/unsloth-gemma-4-E2B-it-loom) | 4.2 GB | tools, reasoning, vision, drafting | 148 vs 109 (+36 %) | 2,811 vs 4,432 (-37 %) | 29/29 |
| `qwen2.5-7b-instruct-loom` | [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/loom-fun/bartowski-Qwen2.5-7B-Instruct-loom) | 4.7 GB | tools | 47 vs 46 (+1 %) | 1,205 vs 826 (+46 %) | 29/29 |
| `qwen2.5-vl-7b-instruct-loom` | [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/loom-fun/ggml-org-Qwen2.5-VL-7B-Instruct-loom) | 6.0 GB | vision | 47 vs 46 (+1 %) | 1,225 vs 807 (+52 %) | 21/28 |
| `gemma-4-e4b-it-loom` | [google/gemma-4-E4B-it](https://huggingface.co/loom-fun/unsloth-gemma-4-E4B-it-loom) | 6.1 GB | tools, reasoning, vision, drafting | 92 vs 75 (+22 %) | 2,003 vs 1,912 (+5 %) | 29/29 |
| `qwen3-8b-loom` | [Qwen/Qwen3-8B](https://huggingface.co/loom-fun/unsloth-Qwen3-8B-loom) | 6.2 GB | tools, reasoning, drafting | 66 vs 52 (+27 %) | 959 vs 656 (+46 %) | 29/29 |
| `qwen3-vl-8b-instruct-loom` | [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/loom-fun/unsloth-Qwen3-VL-8B-Instruct-loom) | 6.2 GB | tools, vision | 45 vs 42 (+7 %) | 958 vs 658 (+46 %) | 29/29 |
| `ornith-1.0-9b-loom` | [deepreinforce-ai/Ornith-1.0-9B](https://huggingface.co/loom-fun/unsloth-Ornith-1.0-9B-loom) | 6.6 GB | tools, reasoning, vision | 41 vs 37 (+10 %) | 1,520 vs 916 (+66 %) | 29/29 |
| `ornith-1.5-9b-loom` | [ornith-ai/Ornith-1.5-9B](https://huggingface.co/loom-fun/bartowski-Ornith-1.5-9B-loom) | 7.6 GB | tools, reasoning, vision, drafting | 62 vs 39 (+59 %) | 1,467 vs 850 (+73 %) | 29/29 |
| `nemotron-3.5-bolt-e4b-loom` | [nemotron-3.5-bolt-e4b](https://huggingface.co/loom-fun/nvidia-NVIDIA-Nemotron-3.5-Bolt-E4B-4GB-loom) | 7.9 GB | tools, reasoning, vision, drafting | 98 vs 55 (+78 %) | 2,387 vs 1,219 (+96 %) | 29/29 |
| `qwen3.5-9b-loom` | [Qwen/Qwen3.5-9B](https://huggingface.co/loom-fun/unsloth-Qwen3.5-9B-loom) | 8.0 GB | tools, reasoning, vision, drafting | 62 vs 36 (+68 %) | 1,484 vs 884 (+68 %) | 29/29 |
| `mellum2.1-12b-a2.5b-thinking-loom` | [JetBrains/Mellum2.1-12B-A2.5B-Thinking](https://huggingface.co/loom-fun/bartowski-Mellum2.1-12B-A2.5B-Thinking-loom) | 8.8 GB | tools, reasoning | 102 vs 76 (+34 %) | 2,928 vs 1,752 (+67 %) | 29/29 |
| `qwen3-14b-loom` | [Qwen/Qwen3-14B](https://huggingface.co/loom-fun/ggml-org-Qwen3-14B-loom) | 9.0 GB | tools, reasoning | 25 vs 24 (+3 %) | 607 vs 331 (+83 %) | 29/29 |
| `gpt-oss-20b-loom` | [openai/gpt-oss-20b](https://huggingface.co/loom-fun/unsloth-gpt-oss-20b-loom) | 11.6 GB | tools, reasoning | 83 vs 78 (+6 %) | 1,934 vs 1,302 (+49 %) | 29/29 |
| `lfm2.5-24b-a2b-loom` | [LiquidAI/LFM2.5-24B-A2B](https://huggingface.co/loom-fun/ggml-org-LFM2.5-24B-A2B-loom) | 14.5 GB | tools, reasoning | 107 vs 99 (+8 %) | 2,918 vs 1,911 (+53 %) | 29/29 |
| `gemma-4-26b-a4b-it-loom` | [google/gemma-4-26B-A4B-it](https://huggingface.co/loom-fun/lmstudio-community-gemma-4-26B-A4B-it-loom) | 17.1 GB | tools, reasoning, drafting | 68 vs 34 (+98 %) | 1,569 vs 917 (+71 %) | 29/29 |
| `qwen3-coder-30b-a3b-instruct-loom` | [Qwen/Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/loom-fun/ggml-org-Qwen3-Coder-30B-A3B-Instruct-loom) | 18.3 GB | tools, drafting | 68 vs 29 (+135 %) | 961 vs 648 (+48 %) | 29/29 |
| `qwen3.8-27b-loom` | [qwen3.8-27b](https://huggingface.co/loom-fun/bartowski-Qwen3.8-27B-loom) | 18.4 GB | tools, reasoning, vision, drafting | 28 vs 23 (+25 %) | 572 vs 309 (+85 %) | 29/29 |
| `qwen3-30b-a3b-instruct-2507-loom` | [Qwen/Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/loom-fun/unsloth-Qwen3-30B-A3B-Instruct-2507-loom) | 18.6 GB | tools, drafting | 86 vs 76 (+12 %) | 931 vs 635 (+47 %) | 29/29 |
| `gemma-4-31b-it-loom` | [google/gemma-4-31B-it](https://huggingface.co/loom-fun/unsloth-gemma-4-31B-it-loom) | 19.3 GB | tools, reasoning, drafting | 24 vs 16 (+50 %) | 315 vs 195 (+61 %) | 29/29 |
| `qwen2.5-32b-instruct-loom` | [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/loom-fun/bartowski-Qwen2.5-32B-Instruct-loom) | 19.9 GB | tools | 12 vs 11 (+3 %) | 308 vs 175 (+75 %) | 29/29 |
| `laguna-xs-2.1-loom` | [poolside/Laguna-XS-2.1](https://huggingface.co/loom-fun/ggml-org-Laguna-XS-2.1-loom) | 20.1 GB | tools, reasoning, drafting | 43 (no llama.cpp run) | 1,475 (no llama.cpp run) | 28/29 |
| `nanbeige4.3-33b-loom` | [Nanbeige/Nanbeige4.3-33B](https://huggingface.co/loom-fun/bartowski-Nanbeige4.3-33B-loom) | 20.8 GB | tools, reasoning | 11 vs 11 (+6 %) | 271 vs 186 (+46 %) | 29/29 |
| `qwen3.6-27b-loom` | [Qwen/Qwen3.6-27B](https://huggingface.co/loom-fun/ggml-org-Qwen3.6-27B-loom) | 21.1 GB | tools, reasoning, vision, drafting | 23 vs 16 (+45 %) | 465 vs 317 (+47 %) | 29/29 |
| `qwen3.6-35b-a3b-loom` | [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/loom-fun/ggml-org-Qwen3.6-35B-A3B-loom) | 21.9 GB | tools, reasoning, vision, drafting | 91 vs 57 (+59 %) | 2,078 vs 1,048 (+98 %) | 29/29 |
| `ornith-1.0-35b-loom` | [ornith-ai/Ornith-1.0-35B](https://huggingface.co/loom-fun/lmstudio-community-Ornith-1.0-35B-loom) | 22.1 GB | tools, reasoning, vision | 75 vs 66 (+14 %) | 2,111 vs 1,058 (+100 %) | 28/29 |
| `minicpm-v-4.7-35b-a3b-loom` | [openbmb/MiniCPM-V-4.7-35B-A3B](https://huggingface.co/loom-fun/bartowski-MiniCPM-V-4.7-35B-A3B-loom) | 22.3 GB | tools, reasoning | 62 vs 53 (+18 %) | 2,159 vs 1,069 (+102 %) | 29/29 |
| `sage-a1.5-loom` | [InternScience/Sage-A1.5](https://huggingface.co/loom-fun/InternScience-Sage-A1.5-loom) | 22.6 GB | tools, reasoning, vision, drafting | 88 vs 71 (+23 %) | 2,103 vs 994 (+112 %) | 29/29 |
| `qwen3.5-35b-a3b-loom` | [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/loom-fun/unsloth-Qwen3.5-35B-A3B-loom) | 23.2 GB | tools, reasoning, vision, drafting | 82 vs 53 (+55 %) | 2,120 vs 1,072 (+98 %) | 29/29 |
| `ornith-1.5-35b-a3b-loom` | [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/loom-fun/bartowski-Ornith-1.5-35B-A3B-loom) | 23.2 GB | tools, reasoning, vision, drafting | 73 vs 39 (+88 %) | 2,062 vs 986 (+109 %) | 29/29 |
| `muse-glimmer-30b-loom` | [meta-models/Muse-Glimmer-30B](https://huggingface.co/loom-fun/ggml-org-Muse-Glimmer-30B-loom) | 24.8 GB | tools, reasoning, vision, drafting | 28 vs 21 (+34 %) | 486 vs 358 (+36 %) | 29/29 |
| `qwen3.8-27b-ud-q4_k_xl-loom` | [qwen3.8-27b-ud-q4_k_xl](https://huggingface.co/loom-fun/unsloth-Qwen3.8-27B-UD-Q4_K_XL) | 18.6 GB | tools, reasoning, vision, drafting | 26 vs 22 (+20 %) | 561 vs 307 (+83 %) | 29/29 |
| `gemma-4-26b-a4b-it-qat-loom` | [google/gemma-4-26B-A4B-it-qat-q4_0-unquantized](https://huggingface.co/loom-fun/unsloth-gemma-4-26B-A4B-it-qat-loom) | 14.5 GB | tools, reasoning, drafting | 102 vs 88 (+16 %) | 1,567 vs 950 (+65 %) | 29/29 |
| `gemma-4-31b-it-qat-loom` | [google/gemma-4-31B-it-qat-q4_0-unquantized](https://huggingface.co/loom-fun/unsloth-gemma-4-31B-it-qat-loom) | 17.6 GB | tools, reasoning, drafting | 31 vs 31 (-1 %) | 309 vs 211 (+46 %) | 29/29 |
| `flint-loom` | [flint](https://huggingface.co/loom-fun/ggml-org-Flint-loom) | 23.6 GB | tools, reasoning, drafting | 111 vs 83 (+34 %) | 1,969 vs 1,077 (+83 %) | 29/29 |
| `inkling-2-mini-loom` | [Inkling-2-Mini](https://huggingface.co/loom-fun/unsloth-Inkling-2-Mini-loom) | 18.1 GB | tools, reasoning, drafting | 43 vs 14 (+214 %) | 1,102 (no llama.cpp run) | 29/29 |
<!-- end models table -->

**Logits:** forced or named `tool_choice`, strict tool schemas, `response_format`, `logprobs`, `min_p` and penalties
sample on the host from the package's logits (every package above returns them; the log says `loom: logits on`).
Those requests run without drafting, so they decode somewhat slower; every other request keeps the device path.

## How it works

Gufo serves a Loom package like any other model: its server, scheduler, prompt cache, parsers and API endpoints are
unchanged. The extension runs the package's own kernels, chat template, tokenizer and drafter through a small C
library (`loom-embed/`) over Loom's text model layer. Details: `DESIGN.md`.

Options: `--loom-device URI`, `--loom-memory-bytes N` (default: none, and Loom's memory pool is committed at load, as
the loom-fun cards serve; N makes it an elastic pool with that budget, which prefills slower on Strix Halo and can refuse
long prompts when small), `--loom-preset NAME` (only for a package directory that holds several presets),
`--loom-logits auto|on|off`. `--cache-disk DIR` keeps the prompt cache across restarts (its checkpoint images need the
elastic pool, which the extension then uses).

## Limits

- In a container, run with `--security-opt seccomp=unconfined` (Loom's GPU memory reservation needs it).
- Loom 0.3.0 reads weights through buffered file I/O by default, which works at a stock locked-memory limit
  (`ulimit -l`: 8 MiB on stock Ubuntu, also Docker's default); the extension no longer falls back to reading whole
  weight files into host memory at load.
- gpt-oss, Glimmer and Inkling 2: automatic tool choice is not grammar-constrained (the package parses the calls); required and named are.

## Quality

Every declared model ran Gufo's API conformance checks (`conformance.py --logits`, 29 checks) on Strix Halo,
all through `gufo model install` and the installed model, 24 on a local machine and 28 on AMD DevLab runners
(`validation/results/models.md`, with Gufo's throughput per model). 45 pass all 29; the other 7 miss checks for the
reasons below.

- gemma-3-1b, HyperCLOVAX-1.5B and Qwen2.5-VL-7B never call tools (their packages declare no tool-call format), so
  the 7 tool checks fail on them.
- Model behavior that loom-serve shows too: Llama-3.2-3B calls the tool again instead of answering; Qwen3.5-2B's
  reasoning ran to the length limit at both low and high effort; Ornith-1.0-35B's reasoning loops when a named
  `tool_choice` conflicts with the question.
- Laguna-XS-2.1 does not reuse a prompt prefix shared by two different conversations (each conversation's own history
  is reused).
- Requests that need logits decode somewhat slower (host sampling, no drafting).

Earlier evidence in `validation/results/`: images, the disk cache tier and the CLI across 48 packages on Loom 0.1.0
(`conformance.md`), Gufo against loom-serve (`bench.md`) and the logits features (`logits.md`).

## Releases

| Version | Gufo | Notes |
|---|---|---|
| 0.4.0 | extension API >= 0.3.35 (gufo-org/gufo main from ec6d3d07) | Loom SDK 0.3.0 (hrx-system v0.3.0, with its C++ source importer); serves 0.3.0 packages and the published 0.2.0 ones (controller ABI 1); weights load at a stock memlock limit |
| 0.3.7 | extension API >= 0.3.35 (gufo-org/gufo main from ec6d3d07) | 52 models: adds Inkling-2-Mini, with a tool grammar for its call syntax (`inkling2`; automatic tool use stays on the device) |
| 0.3.6 | extension API >= 0.3.35 (gufo-org/gufo main from ec6d3d07) | 51 models: adds Qwen3.8-27B UD-Q4_K_XL, gemma-4 26B-A4B and 31B QAT, and Flint |
| 0.3.5 | extension API >= 0.3.35 (gufo-org/gufo main from ec6d3d07) | Validation: all 47 models through `gufo model install` (`validation/results/models.md`) |
| 0.3.4 | extension API >= 0.3.35 (gufo-org/gufo main from ec6d3d07) | Loom SDK release checks: a package built for another SDK is refused at load with what to fetch; `tools/models.py` pins each card's revision for the runtime's SDK (its `v<sdk>` tag, else the default branch) and `--check` verifies the pins |
| 0.3.3 | extension API >= 0.3.35 (gufo-org/gufo main from ec6d3d07) | Every package pinned at its MIT-licensed card revision (the package files are unchanged) |
| 0.3.2 | extension API >= 0.3.35 (gufo-org/gufo main from ec6d3d07) | 47 models: gemma-4-12B-it is left out until its drafter (`Anbeeld/gemma4-12B-it-DFlash-GGUF`) declares a license; licenses the Hub knows as tags are read; `tests/build_test.sh` builds with Gufo main's Rust crates (tested on 029a156) |
| 0.3.1 | extension API >= 0.3.35 (gufo-org/gufo main from ec6d3d07) | Every model's package pinned at the revision that declares its license (the package files are unchanged) |
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
- A new Loom SDK release (`DESIGN.md`, Loom SDK releases): bump `nix/loom-runtime.nix` and
  `LOOM_TEXT_EMBED_HRX_LOOM`, rerun `tools/models.py` (it pins each card's revision for that SDK and refuses others;
  `--check` verifies existing pins), lock, validate and release.
- hrx-system pin: after changing it in `nix/loom-runtime.nix`, regenerate its downloads with
  `python3 nix/hrx-deps.py generate <hrx-system checkout> nix/hrx-deps.nix`.
- `nix flake lock --update-input gufo` moves the flake to a newer Gufo.
