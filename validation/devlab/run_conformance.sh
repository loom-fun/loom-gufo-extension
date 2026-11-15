#!/usr/bin/env bash
# One package through the logits checks on a runner's GPU, with a bundle from build_bundle.sh unpacked at $BUNDLE.
#   BUNDLE=DIR run_conformance.sh PACKAGE_DIR OUT_DIR [PORT]
# Writes OUT_DIR/conformance.json (conformance.py --logits), OUT_DIR/logits_check.json (host == device greedy,
# logprob consistency) and OUT_DIR/gufo.log. Exit status: nonzero when a check failed.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
BUNDLE=${BUNDLE:?unpacked bundle directory}
pkg=$(cd "${1:?PACKAGE_DIR}" && pwd); out=$(mkdir -p "${2:?OUT_DIR}" && cd "$2" && pwd); port=${3:-8181}
log=$out/gufo.log
"$BUNDLE/run-gufo.sh" serve llm --model "$pkg" --sessions 4 --port "$port" --host 127.0.0.1 --loom-logits=on ${EXTRA_GUFO_ARGS:-} > "$log" 2>&1 &
pid=$!
trap 'kill $pid 2>/dev/null; wait $pid 2>/dev/null' EXIT
for _ in $(seq 1 600); do   # up to 20 minutes: loading and the package's JIT
  curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1 && break
  kill -0 $pid 2>/dev/null || { echo "gufo exited while loading"; tail -20 "$log"; exit 2; }
  sleep 2
done
grep -m1 "loom: logits on" "$log" || { echo "the package did not load with logits"; tail -5 "$log"; exit 3; }
rc=0
python3 "$BUNDLE/validation/conformance.py" "http://127.0.0.1:$port" "$out/conformance.json" --logits > "$out/conformance.log" 2>&1 || rc=$?
python3 "$BUNDLE/validation/devlab/logits_check.py" "http://127.0.0.1:$port" "$out/logits_check.json" > "$out/logits_check.log" 2>&1 || rc=$?
tail -3 "$out/conformance.log"; tail -3 "$out/logits_check.log"
exit $rc
