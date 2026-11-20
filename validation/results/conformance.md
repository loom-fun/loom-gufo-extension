# Gufo conformance of loom-fun packages (35B parameters or less)

## Module install on Gufo main (2026-11-20)

Strix Halo (gfx1151), Gufo main ec6d3d07 (`nix build github:gufo-org/gufo/<rev>#sdk`, extension API 0.3.35), the
extension installed as a module with `gufo extension install` (library from `nix/runtime.nix`, hrx-system v0.2.0),
loom-fun packages fetched with `loom-fetch huggingface:loom-fun/<repo>:chat` (hrx-loom 0.2.0). Default options,
`--sessions 4`, a stock 8 MiB locked-memory limit.

| Package | conformance.py --logits (29 checks) |
|---|---|
| unsloth-Qwen3-0.6B-loom | 29 pass |
| bartowski-Qwen2.5-0.5B-Instruct-loom | 29 pass |
| ggml-org-gemma-3-1b-it-loom | 21 pass, 7 fail: the tool checks, with the tool round trip skipped (the package declares no tool-call format; see below) |

## Loom 0.2.0 (2026-11-17)

Strix Halo (gfx1151), hrx-loom 0.2.0 (hrx-system v0.2.0, `nix/runtime.nix`), Gufo at gufo-org/gufo#1243 (2255d6f3),
the 0.2.0 packages from loom-fun-staging (preset `chat`, the same files the loom-fun promotion publishes). Default
options (fixed pool), `--sessions 4`, a stock 8 MiB locked-memory limit (weights preloaded into host memory).

| Package | conformance.py --logits (29 checks) |
|---|---|
| unsloth-Qwen3.5-4B-loom | 29 pass |
| bartowski-LFM3-1.7B-loom | 29 pass |
| unsloth-gemma-4-E2B-it-loom | 29 pass |
| unsloth-gpt-oss-20b-loom | 29 pass |
| ggml-org-LFM2.5-24B-A2B-loom | 29 pass |
| ggml-org-Muse-Glimmer-30B-loom | 29 pass |
| unsloth-Qwen3-VL-4B-Instruct-loom | 29 pass |

- Unlimited locked memory (weights streamed from disk): Qwen3.5-4B 29 pass.
- Vision (`vision_test.py`): Qwen3-VL-4B pass.
- Disk tier across a restart (`--cache-disk`, elastic pool): Qwen3.5-4B and LFM3-1.7B restore the prompt from disk
  with the same greedy answer.
- Speed (`bench.py`, Qwen3.5-4B, 2.2k-token prefill): decode 88.0 tok/s, prefill 2666 tok/s, 4 streams 215.8 tok/s
  (Loom 0.1.0 through Gufo: 86.3 / 2521 / 208.9). An elastic pool (`--loom-memory-bytes 48000000000`): 88.3 / 2336 /
  210.1, hence the fixed default.

## Loom 0.1.0 (2026-11-14)

Run 2026-11-14 on Strix Halo (gfx1151), hrx-loom 0.1.0: 48 packages,
`conformance.py` (27 Gufo checks each), `vision_test.py` (20 vision packages),
a `gufo prompt` CLI check, and `disk_check.py` (disk tier across a restart).

- 1,134 checks pass, 14 fail, 145 are gaps that need logits from Loom
  (3 tool round-trip checks are skipped when the first tool call fails).
- All 14 failures are model behavior; loom-serve shows the same:
  gemma-3-1b, HyperCLOVAX-1.5B and Qwen2.5-VL-7B never call tools (4 checks each);
  Llama-3.2-3B calls the tool again instead of answering; Qwen3.5-0.8B's reasoning
  loops to the length limit.
- Gaps on every package: tool_choice required, min_p/penalties, logprobs, plus one
  response_format gap (it is followed by prompting, not enforced, everywhere).
- Reasoning token usage and reasoning effort: 48/48.
- Disk tier (`--cache-disk`): 48/48 restore the prompt from disk after a restart
  with the same greedy answer.
- Vision: 20/20. CLI: 48/48.
- Greedy parity with loom-serve: 39/48 identical on 3/3 prompts; 9 split at a
  near-tie on one prompt, within Loom's batch-shape variation (loom-serve gives
  gufo's Qwen3-8B haiku itself under load).
