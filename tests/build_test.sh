#!/bin/bash
# Builds and runs the Loom runner's host-sampling tests against a fake
# libloom_text_embed. No GPU, no Loom runtime. Needs a Gufo checkout (the
# loom-extension branch) and g++ 13+.
#   GUFO=/path/to/gufo ./build_test.sh [OUT_DIR]
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
GUFO=${GUFO:?set GUFO to a gufo checkout}
OUT=${1:-${TMPDIR:-/tmp}/loom_runner_test}
mkdir -p "$OUT/gufo/ext/generated"
# Gufo's cache and registry libraries are not linked: the tests never reach them
# (unresolved symbols are ignored at link time).
# The generated option/provider constants (the real ones come from extension.json).
cat > "$OUT/gufo/ext/generated/loom_constants.hpp" <<'EOF'
#pragma once
#include <gufo/ext/registry.hpp>
namespace gufo::ext::generated::extension_loom { inline const ProviderInfo kProvider_loom_text{}; }
EOF
mkdir -p "$OUT/gufo/ext/detail"
cat > "$OUT/gufo/ext/detail/api_version.hpp" <<'EOF'
#pragma once
#include <string_view>
namespace gufo::ext {
inline constexpr unsigned kApiMajor = 0, kApiMinor = 3, kApiRevision = 15;
inline constexpr std::string_view kApiVersion = "0.3.15";
}  // namespace gufo::ext
EOF
CXX=${CXX:-g++}
FLAGS="-std=c++20 -O1 -g -I$OUT -I$GUFO -I$GUFO/include -I$here/../loom-embed"
objs=()
for f in src/core/sampling.cpp src/core/sampling_logit_row.cpp src/core/json_constraint.cpp \
         src/core/json_schema_lexeme.cpp src/core/json_schema_regex.cpp src/ext/text.cpp src/core/crypto/sha256.cpp; do
  o="$OUT/$(basename "$f" .cpp).o"; objs+=("$o")
  [ "$o" -nt "$GUFO/$f" ] || $CXX $FLAGS -c "$GUFO/$f" -o "$o" &
done
wait
$CXX $FLAGS -c "$here/../src/loom_chat.cpp" -o "$OUT/loom_chat.o" &
$CXX $FLAGS -c "$here/fake_embed.cpp" -o "$OUT/fake_embed.o" &
$CXX $FLAGS -c "$here/runner_logits_test.cpp" -o "$OUT/runner_logits_test.o" &
wait
$CXX "${objs[@]}" "$OUT/fake_embed.o" "$OUT/loom_chat.o" "$OUT/runner_logits_test.o" -o "$OUT/runner_logits_test" -lpthread -lcrypto -Wl,--unresolved-symbols=ignore-all $(pkg-config --libs icu-uc icu-i18n 2>/dev/null || echo "-licuuc -licui18n")
"$OUT/runner_logits_test"
