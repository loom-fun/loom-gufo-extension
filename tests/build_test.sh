#!/bin/bash
# Builds and runs the Loom runner's host-sampling tests against a fake
# libloom_text_embed. No GPU, no Loom runtime. Needs a Gufo checkout (main at
# ec6d3d07 or later), g++ 13+ and Cargo.
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
inline constexpr unsigned kApiMajor = 0, kApiMinor = 3, kApiRevision = 35;
inline constexpr std::string_view kApiVersion = "0.3.35";
}  // namespace gufo::ext
EOF
# The embed's SDK release (its header) is the one nix/loom-runtime.nix builds it from.
header_sdk=$(sed -n 's/^#define LOOM_TEXT_EMBED_HRX_LOOM "\(.*\)"/\1/p' "$here/../loom-embed/loom_text_embed.h")
nix_sdk=$(sed -n 's/^  version = "\(.*\)";/\1/p' "$here/../nix/loom-runtime.nix")
[ -n "$header_sdk" ] && [ "$header_sdk" = "$nix_sdk" ] || { echo "LOOM_TEXT_EMBED_HRX_LOOM ($header_sdk) != nix/loom-runtime.nix version ($nix_sdk)"; exit 1; }
CXX=${CXX:-g++}
# Gufo's Rust crates (src/core/*/include) hold the C headers of the code they replaced.
RUST_INCLUDES=$(for d in "$GUFO"/src/core/*/include; do printf -- ' -I%s' "$d"; done)
FLAGS="-std=c++20 -O1 -g -I$OUT -I$GUFO -I$GUFO/include$RUST_INCLUDES -I$here/../loom-embed"
objs=()
for f in src/core/sampling.cpp src/core/sampling_logit_row.cpp src/core/json_constraint.cpp \
         src/core/json_schema_lexeme.cpp src/core/json_schema_regex.cpp src/ext/text.cpp src/core/crypto/sha256.cpp \
         src/core/json.cpp; do
  o="$OUT/$(basename "$f" .cpp).o"; objs+=("$o")
  [ "$o" -nt "$GUFO/$f" ] || $CXX $FLAGS -c "$GUFO/$f" -o "$o" &
done
# Gufo parses JSON in Rust (src/core/json, linked through the gufo-rust static library).
cargo build --quiet --release -p gufo-rust --manifest-path "$GUFO/Cargo.toml" --target-dir "$OUT/cargo" &
wait
$CXX $FLAGS -c "$here/../src/loom_chat.cpp" -o "$OUT/loom_chat.o" &
$CXX $FLAGS -c "$here/fake_embed.cpp" -o "$OUT/fake_embed.o" &
$CXX $FLAGS -c "$here/runner_logits_test.cpp" -o "$OUT/runner_logits_test.o" &
$CXX $FLAGS -c "$here/package_test.cpp" -o "$OUT/package_test.o" &
wait
$CXX "${objs[@]}" "$OUT/fake_embed.o" "$OUT/loom_chat.o" "$OUT/runner_logits_test.o" -o "$OUT/runner_logits_test" "$OUT/cargo/release/libgufo_rust.a" -lpthread -ldl -lcrypto -Wl,--unresolved-symbols=ignore-all $(pkg-config --libs icu-uc icu-i18n 2>/dev/null || echo "-licuuc -licui18n")
"$OUT/runner_logits_test"
$CXX "$OUT/package_test.o" "$OUT/json.o" -o "$OUT/package_test" "$OUT/cargo/release/libgufo_rust.a" -lpthread -ldl
"$OUT/package_test"
