# Gufo conformance of loom-fun packages (35B parameters or less)

Run 2026-11-14 on Strix Halo (gfx1151): 48 packages, `conformance.py` (25 gufo
checks each), `vision_test.py` (20 vision packages) and a `gufo prompt` CLI check.
Raw results: `conf4/`.

- 1,038 checks pass, 14 fail, 145 are gaps that need logits (3 round-trip checks are skipped when the first tool call fails).
- All 14 failures are model behavior; loom-serve shows the same (`parity.log`):
  gemma-3-1b, HyperCLOVAX-1.5B and Qwen2.5-VL-7B never call tools (4 checks each);
  Llama-3.2-3B calls the tool again instead of answering; Qwen3.5-0.8B's reasoning
  loops to the length limit.
- Gaps on every package: tool_choice required, min_p/penalties, logprobs, plus one response_format
  gap (it is followed by prompting, not enforced, everywhere).
- Vision: 20/20 pass. CLI: 48/48 answer.
- Greedy parity with loom-serve: 39/48 identical on 3/3 prompts; 9 split at a
  near-tie on one prompt, within Loom's batch-shape variation (loom-serve gives
  gufo's Qwen3-8B haiku itself under load).
