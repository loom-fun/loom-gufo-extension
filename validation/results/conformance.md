# Gufo conformance of loom-fun packages (35B parameters or less)

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
