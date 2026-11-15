# Running the logits checks on a GPU runner (Strix Halo and Radeon AI PRO R9700)

How the Gufo logits checks run on a CI runner that has a GPU but no Nix or Docker (AMD DevLab Dispatch). The Nix change that builds
the embed library for both GPUs is in loom-fun/gufo (`gpuTargets`, branch `logits-targets`).

## What runs where
- **No Nix or Docker on the runner.** Everything is built on a dev box and shipped as one tarball (the bundle): a CPU-only
  `gufo` with the loom extension, `libloom_text_embed.so` (Loom's text model layer, built inside hrx-system v0.1.0 for
  `gfx1151` and `gfx1201`), every shared library they need and the loader. `run.sh` starts a program through the bundled loader
  and libraries, so the runner contributes only the kernel (amdgpu driver, `/dev/kfd`, `/dev/dri/renderD*`) and Python 3.
  The model work is Loom's (device code inside the embed library); gufo itself runs without HIP.
- **Two programs, two levels.**
  1. `run.sh embed_probe PACKAGE [STEPS] [ROWS]` (no Gufo, seconds): opens the package with logits on and ROWS resident rows,
     decodes all rows in one epoch per step and checks for every row and step that `argmax(exported row) == the token the device
     selected` (greedy request), that padding is never the argmax, that feeding the host's token (`decode_batch_tokens`) and the
     device's own (`decode_batch`) alternately agree, and that a request without `"logits": true` leaves no row. Prints one JSON
     line, exit 0 only if every check held. This is the kit's on-GPU proof (slot indexing, export kernel, allocation layout).
  2. `validation/devlab/run_conformance.sh PACKAGE OUT` (gufo): serves the package with `--loom-logits=on`, runs
     `validation/conformance.py --logits` (29 checks: tool_choice required and named, strict tool schema, response_format
     json_schema, min_p and penalties actually applied, logprobs consistent: all must PASS, not gap) and
     `validation/devlab/logits_check.py` (host greedy == device greedy on the same server, logprob consistency, grammar and plain
     requests batched together). Writes `conformance.json`, `logits_check.json`, `gufo.log` in OUT.
- Order: `embed_probe` first (cheap, isolates package/embed faults), then the Gufo run only if it passed.

## Build the bundle (dev box, once per change of the embed, the runner or the extension)
```sh
git clone -b logits-readback https://github.com/loom-fun/loom-gufo-extension ext
git clone -b logits-targets  https://github.com/loom-fun/gufo gufo          # base loom-extension + gpuTargets
ext/validation/devlab/build_bundle.sh gufo OUT_DIR "gfx1151 gfx1201"
# -> OUT_DIR/gufo-loom-<gufo rev>-<extension rev>.tar.gz (+ .sha256), ~300 MB
```
First run builds ROCm's LLVM from source inside the `nixos/nix` image (about 40 minutes on 24 cores, ~60 GB of disk in the docker
volume `loom-bundle-nix`); later runs reuse the volume (embed + gufo: about 15 minutes). The bundle runs anywhere with a Linux
x86-64 kernel and the amdgpu driver; `run.sh gufo --version` works without a GPU.

## Ship it to the runner
The runner job needs a URL. Attach the tarball and its `.sha256` to a prerelease of loom-fun/loom-gufo-extension (tag
`devlab-bundle-<rev>`); the job downloads it with `curl -fsSL` (GitHub release assets are public; `GH_TOKEN` avoids rate limits)
and verifies the sha256 before unpacking. Pin the sha256 in the devlab config like `data/llamacpp-vulkan.sha256` pins its archive.

## The `logits:V` / `gufo:V` step (devlab_perf.sh), in shell
```sh
# V = the variant's package directory (built with --build from the kit branch; its model.json declares "llm.logits": false)
B=$WORK/gufo-bundle
if [ ! -x "$B/run.sh" ]; then
  mkdir -p "$B" && curl -fsSL --retry 5 -o "$WORK/bundle.tgz" "$BUNDLE_URL" \
    && echo "$BUNDLE_SHA256  $WORK/bundle.tgz" | sha256sum -c - && tar xzf "$WORK/bundle.tgz" -C "$B" --strip-components=1
fi
"$SV/stop.sh" ...                          # the job's loom-serve must be stopped: one process owns the GPU memory budget
"$B/run-embed-probe.sh" "$PKG" 24 3 | tee "$out/logits_$V.json"            # logits:V
BUNDLE=$B "$B/validation/devlab/run_conformance.sh" "$PKG" "$out/gufo_$V"  # gufo:V
```
- Budget: `embed_probe` loads the package once (JIT of its programs, as loom-serve does); the Gufo run loads it again with 4 rows.
  Both use the package's own `memory_bytes` budget logic; pass `--loom-memory-bytes N` to `gufo serve` on a 32 GB R9700 if the
  load would not fit next to nothing else (`run_conformance.sh` forwards `EXTRA_GUFO_ARGS`).
- Time: embed_probe 1-3 minutes per package; the conformance run 5-15 minutes (reasoning checks dominate; models that always reason
  need the longer end).
- `make_info.py` fields to read: `embed_probe.failures == 0` (kit proof: per GPU), `conformance.json` counts with `gap == 0 and
  fail == 0` for the logits checks, `logits_check.json` all pass. Card text: "host logits: supported (run link)" per GPU.

## What a failure means
| Symptom | Likely cause |
|---|---|
| `open: ... does not declare llm.logits` | package built before the kit change (no key in `model.json`) |
| `open: ... has no logits region` | allocation layout differs from the embed's reader (offset 32768, slots = min(spans, rows)): kit/embed disagree |
| embed_probe: `argmax(row)` != device token at some step/row | export kernel or slot index wrong (look at the first failing step/row: row 0 failing = kernel, row k only = slot stride) |
| embed_probe: argmax is padding | `vocab` of the head vs the row width, or a stale row (export skipped) |
| conformance `tool_choice required` fail, others pass | the package's native tool-call dialect is not Gufo's JSON one (gufo.log: `native tool calls differ`): expected for XML/pythonic dialects, grammars stay off for that package |
| conformance reasoning checks fail with `--logits` only | the package's raw reasoning markers are not `<think>`; grammars should have stayed off (`reasoning markers differ` in gufo.log): runner bug if it did not |
| everything slow | a logits load keeps `+min(spans, rows) x vocab x 4` bytes; `--loom-logits=off` is the control for any timing question |

## Not verified here (no GPU in this environment)
The bundle builds and `run.sh` executes on the dev box; the embed's download path, the runner against real logits and
`embed_probe` have not run on a GPU. The runner's host-sampling logic is covered by the fake-embed tests
(`tests/build_test.sh`, 11 cases), the embed by compiling and linking in the pinned Loom tree.
