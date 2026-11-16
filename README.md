# loom-gufo-extension

Runs [loom-fun](https://huggingface.co/loom-fun) Loom model packages in [Gufo](https://github.com/gufo-org/gufo) on
Strix Halo as ordinary Gufo text models: Gufo's server, API endpoints, prompt cache and parsers, with the package's
own Loom kernels doing the model work. It builds against the `loom-extension` branch of
[loom-fun/gufo](https://github.com/loom-fun/gufo/tree/loom-extension), a fork of Gufo that adds model-directory support.

## Build

Needs Linux x86-64 and [Nix](https://nixos.org/download/) with flakes. Gufo's binary cache avoids building ROCm's LLVM
from source (hours): put these lines in `/etc/nix/nix.conf` (and restart the Nix daemon), or in `NIX_CONFIG`:

```
experimental-features = nix-command flakes
extra-substituters = https://gufo.cachix.org
extra-trusted-public-keys = gufo.cachix.org-1:rfZe3n4lHINgyoACKzs5EthIZmBNvEyCCtufUpth0PE=
```

Then, in an empty directory (a few minutes with the cache):

```sh
git clone -b loom-extension https://github.com/loom-fun/gufo
git clone https://github.com/loom-fun/loom-gufo-extension
cd gufo
LOOM=$(nix build --impure --no-link --print-out-paths --expr \
  "((builtins.getFlake \"path:$PWD\").packages.x86_64-linux.loom-runtime.override { embed = $PWD/../loom-gufo-extension/loom-embed; })")
nix develop --impure --expr "(builtins.getFlake \"path:$PWD\").packages.x86_64-linux.default" --command bash -c "
  export PKG_CONFIG_PATH=$LOOM/lib/pkgconfig:\$PKG_CONFIG_PATH && eval \"\$preConfigure\" &&
  cmake -S . -B build -G Ninja \$cmakeFlags -DENGINE_ENABLE_HIP=ON -DGUFO_EXTENSIONS=loom \
    -DGUFO_EXTENSION_DIRS=$PWD/../loom-gufo-extension &&
  cmake --build build --target gufo"
build/gufo diagnose --section extensions     # lists "provider loom-text: available"
cd ..
```

No Nix on the machine (or no root)? Run the same commands in the `nixos/nix` image, keeping `/nix` in a named volume so
the built binary's libraries are still there when you serve:

```sh
docker run --rm -it -v gufo-nix:/nix -v $PWD:$PWD -w $PWD -e NIX_CONFIG="$(printf '%s\n' \
  'experimental-features = nix-command flakes' 'extra-substituters = https://gufo.cachix.org' \
  'extra-trusted-public-keys = gufo.cachix.org-1:rfZe3n4lHINgyoACKzs5EthIZmBNvEyCCtufUpth0PE=')" nixos/nix bash
```

To serve from it, add `--device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined --network host`, and mount
the Hugging Face cache at the same path (see Quick start).

## Quick start

On Strix Halo (gfx1151) with the amdgpu driver, from the directory where you ran Build (it holds `gufo/`):

```sh
pipx install 'hrx-loom[hf]==0.1.0'     # loom-fetch (and loom-serve); pipx: `sudo apt install pipx` or `python3 -m pip install --user pipx`
loom-fetch https://huggingface.co/loom-fun/ggml-org-Qwen3-4B-loom/resolve/main/model.json --output=models/qwen3-4b --no-input
gufo/build/gufo serve llm --model models/qwen3-4b --sessions 4     # listens on 127.0.0.1:8080 (--host, --port)
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-4B","messages":[{"role":"user","content":"Hello!"}]}'
```

Any model in the table below works the same way: `loom-fetch https://huggingface.co/loom-fun/<repo>/resolve/main/model.json
--output=<dir> --no-input`. The request's `model` is the name in the table (`GET /v1/models` lists it). `loom-fetch`
links the weights from the Hugging Face cache (`~/.cache/huggingface`), so the cache must stay where it was (and be
mounted at the same path in a container); a package takes about its GGUF's size on disk.

**Logits (until the next loom-fun release):** forced or named `tool_choice`, strict tool schemas, `response_format`,
`logprobs`, `min_p` and penalties need packages that return logits. Those are in
[loom-fun-staging](https://huggingface.co/loom-fun-staging) (the same repos as the table) until they are promoted to
loom-fun. They carry two presets, so name the one for your GPU (`chat` is Strix Halo, `chat-r9700` the Radeon AI PRO
R9700); Gufo serves the fetched preset without any flag:

```sh
loom-fetch https://huggingface.co/loom-fun-staging/unsloth-gpt-oss-20b-loom/resolve/main/model.json --preset chat \
  --output=models/gpt-oss-20b --no-input
```

The server log then says `loom: logits on (...)`. Without logits those features are not enforced: a forced
`tool_choice` returns an error (`tool_choice_unsatisfied`) when the model does not call, `response_format` is only
asked of the model, and `logprobs`, `min_p` and penalties are ignored.

## Models

The loom-fun packages of 35B parameters or less, on Strix Halo. Loom: through Gufo with this extension, a 32,000-token
prompt for pp32k and single-stream decode after a short prompt. llama.cpp (ROCm b1374): the package card's baseline.

| Model | Loom pp32k (tok/s) | llama.cpp pp32k (tok/s) | Loom decode (tok/s) | llama.cpp decode (tok/s) |
|---|---|---|---|---|
| [Qwen/Qwen3-0.6B](https://huggingface.co/loom-fun/unsloth-Qwen3-0.6B-loom) | 3088 | 2717 | 340.0 | 266.4 |
| [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/loom-fun/bartowski-Qwen2.5-0.5B-Instruct-loom) | 7451 | 6501 | 360.9 | 293.4 |
| [Qwen/Qwen3.5-0.8B](https://huggingface.co/loom-fun/unsloth-Qwen3.5-0.8B-loom) | 8011 | 5639 | 260.5 | 202.0 |
| [google/gemma-3-1b-it](https://huggingface.co/loom-fun/ggml-org-gemma-3-1b-it-loom) | 5690 | — | 204.9 | 168.3 |
| [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/loom-fun/bartowski-Qwen2.5-1.5B-Instruct-loom) | 2470 | 2576 | 173.4 | 156.0 |
| [LiquidAI/LFM3-Omni-2B](https://huggingface.co/loom-fun/ggml-org-LFM3-Omni-2B-loom) | 5357 | 4330 | 165.7 | 148.1 |
| [LiquidAI/LFM3-1.7B](https://huggingface.co/loom-fun/bartowski-LFM3-1.7B-loom) | 5313 | 4128 | 166.0 | 137.4 |
| [naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B](https://huggingface.co/loom-fun/rippertnt-HyperCLOVAX-SEED-Text-Instruct-1.5B-loom) | 2859 | 2308 | 174.4 | 155.6 |
| [Qwen/Qwen3-1.7B](https://huggingface.co/loom-fun/ggml-org-Qwen3-1.7B-loom) | 2470 | 2030 | 159.8 | 144.0 |
| [Qwen/Qwen3.5-2B](https://huggingface.co/loom-fun/unsloth-Qwen3.5-2B-loom) | 5093 | 3489 | 139.8 | 118.3 |
| [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/loom-fun/unsloth-Llama-3.2-3B-Instruct-loom) | 1333 | 1075 | 94.7 | 90.2 |
| [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/loom-fun/unsloth-Qwen3-4B-Instruct-2507-loom) | 1095 | 867 | 75.3 | 73.5 |
| [XHToken/Spark-X2.5-4B](https://huggingface.co/loom-fun/XHToken-Spark-X2.5-4B-loom) | 2068 | 1274 | 70.3 | 61.1 |
| [XHToken/Spark-X2.6-4B](https://huggingface.co/loom-fun/bartowski-Spark-X2.6-4B-loom) | 2049 | 1340 | 69.7 | 61.5 |
| [Qwen/Qwen3-4B](https://huggingface.co/loom-fun/ggml-org-Qwen3-4B-loom) | 1078 | 868 | 102.2 | 74.1 |
| [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/loom-fun/unsloth-Qwen3-VL-4B-Instruct-loom) | 1097 | 838 | 75.1 | 71.1 |
| [Qwen/Qwen3.5-4B](https://huggingface.co/loom-fun/unsloth-Qwen3.5-4B-loom) | 2202 | 1353 | 85.6 | 55.9 |
| [google/gemma-4-E2B-it](https://huggingface.co/loom-fun/unsloth-gemma-4-E2B-it-loom) | 2585 | 4432 | 148.5 | 108.6 |
| [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/loom-fun/bartowski-Qwen2.5-7B-Instruct-loom) | 1120 | 826 | 46.1 | 46.4 |
| [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/loom-fun/ggml-org-Qwen2.5-VL-7B-Instruct-loom) | 1113 | 807 | 46.0 | 46.2 |
| [google/gemma-4-E4B-it](https://huggingface.co/loom-fun/unsloth-gemma-4-E4B-it-loom) | 1892 | 1912 | 86.7 | 75.4 |
| [Qwen/Qwen3-8B](https://huggingface.co/loom-fun/unsloth-Qwen3-8B-loom) | 846 | 656 | 49.7 | 52.3 |
| [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/loom-fun/unsloth-Qwen3-VL-8B-Instruct-loom) | 869 | 658 | 40.6 | 42.0 |
| [deepreinforce-ai/Ornith-1.0-9B](https://huggingface.co/loom-fun/unsloth-Ornith-1.0-9B-loom) | 1439 | 916 | 36.5 | 37.2 |
| [google/gemma-4-12B-it](https://huggingface.co/loom-fun/unsloth-gemma-4-12B-it-loom) | 702 | 525 | 42.9 | 29.7 |
| [ornith-ai/Ornith-1.5-9B](https://huggingface.co/loom-fun/bartowski-Ornith-1.5-9B-loom) | 1394 | 850 | 51.2 | 38.9 |
| [nemotron-3.5-bolt-e4b](https://huggingface.co/loom-fun/nvidia-NVIDIA-Nemotron-3.5-Bolt-E4B-4GB-loom) | 1867 | 1219 | 81.1 | 54.9 |
| [Qwen/Qwen3.5-9B](https://huggingface.co/loom-fun/unsloth-Qwen3.5-9B-loom) | 1398 | 884 | 53.9 | 36.5 |
| [JetBrains/Mellum2.1-12B-A2.5B-Thinking](https://huggingface.co/loom-fun/bartowski-Mellum2.1-12B-A2.5B-Thinking-loom) | 2762 | 1752 | 95.2 | 76.0 |
| [Qwen/Qwen3-14B](https://huggingface.co/loom-fun/ggml-org-Qwen3-14B-loom) | 544 | 331 | 24.7 | 24.5 |
| [openai/gpt-oss-20b](https://huggingface.co/loom-fun/unsloth-gpt-oss-20b-loom) | 1839 | 1302 | 79.6 | 78.1 |
| [LiquidAI/LFM2.5-24B-A2B](https://huggingface.co/loom-fun/ggml-org-LFM2.5-24B-A2B-loom) | 2842 | 1911 | 104.8 | 99.2 |
| [google/gemma-4-26B-A4B-it](https://huggingface.co/loom-fun/lmstudio-community-gemma-4-26B-A4B-it-loom) | 1395 | 917 | 62.2 | 34.2 |
| [Qwen/Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/loom-fun/ggml-org-Qwen3-Coder-30B-A3B-Instruct-loom) | 899 | 648 | 64.0 | 28.9 |
| [qwen3.8-27b](https://huggingface.co/loom-fun/bartowski-Qwen3.8-27B-loom) | 442 | 294 | 21.1 | 21.1 |
| [Qwen/Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/loom-fun/unsloth-Qwen3-30B-A3B-Instruct-2507-loom) | 911 | 635 | 84.3 | 76.5 |
| [google/gemma-4-31B-it](https://huggingface.co/loom-fun/unsloth-gemma-4-31B-it-loom) | 282 | 195 | 20.2 | 16.1 |
| [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/loom-fun/bartowski-Qwen2.5-32B-Instruct-loom) | 274 | 175 | 11.2 | 11.2 |
| [poolside/Laguna-XS-2.1](https://huggingface.co/loom-fun/ggml-org-Laguna-XS-2.1-loom) | 1376 | 636 | 41.1 | 49.5 |
| [Nanbeige/Nanbeige4.3-33B](https://huggingface.co/loom-fun/bartowski-Nanbeige4.3-33B-loom) | 259 | 186 | 11.0 | 10.6 |
| [Qwen/Qwen3.6-27B](https://huggingface.co/loom-fun/ggml-org-Qwen3.6-27B-loom) | 437 | 317 | 21.8 | 15.8 |
| [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/loom-fun/ggml-org-Qwen3.6-35B-A3B-loom) | 1947 | 1048 | 74.8 | 57.1 |
| [ornith-ai/Ornith-1.0-35B](https://huggingface.co/loom-fun/lmstudio-community-Ornith-1.0-35B-loom) | 1982 | 1058 | 72.5 | 65.7 |
| [openbmb/MiniCPM-V-4.7-35B-A3B](https://huggingface.co/loom-fun/bartowski-MiniCPM-V-4.7-35B-A3B-loom) | 1992 | 1069 | 59.7 | 52.6 |
| [InternScience/Sage-A1.5](https://huggingface.co/loom-fun/InternScience-Sage-A1.5-loom) | 1976 | 994 | 69.7 | 71.2 |
| [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/loom-fun/unsloth-Qwen3.5-35B-A3B-loom) | 1952 | 1072 | 73.7 | 53.1 |
| [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/loom-fun/bartowski-Ornith-1.5-35B-A3B-loom) | 1937 | 986 | 63.5 | 38.8 |
| [meta-models/Muse-Glimmer-30B](https://huggingface.co/loom-fun/ggml-org-Muse-Glimmer-30B-loom) | 444 | 358 | 29.2 | 21.0 |

## How it works

Gufo serves a Loom package like any other model: its server, scheduler, prompt cache, parsers and API endpoints are
unchanged. The extension runs the package's own kernels, chat template, tokenizer and drafter through a small C
library (`loom-embed/`) over Loom's text model layer. Details: `DESIGN.md`.

Options: `--loom-device URI`, `--loom-memory-bytes N` (default: no limit; a small budget can refuse long prompts),
`--loom-preset NAME` (only for a package directory that holds several presets), `--loom-logits auto|on|off`.
`--cache-disk DIR` keeps the prompt cache across restarts.

## Logits

Requests that need logits (see Quick start) sample on the host and run without drafting, so they decode somewhat
slower; every other request keeps the device path.

## Limits

- In a container, run with `--security-opt seccomp=unconfined` (Loom's GPU memory reservation needs it).
- gpt-oss and Glimmer: automatic tool choice is not grammar-constrained (the package parses the calls); required and named are.

## Development

Runner tests (no GPU, against a fake `libloom_text_embed`): `GUFO=<gufo checkout> tests/build_test.sh`.
