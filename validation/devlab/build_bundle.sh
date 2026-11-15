#!/usr/bin/env bash
# Builds the DevLab bundle: a CPU-only gufo with the loom extension plus the
# libloom_text_embed.so it links, as one relocatable tarball (no GPU needed to
# build; the GPU work is Loom's, in the library). Run on a dev box with Docker;
# the toolchain is Nix inside the nixos/nix image, so nothing is installed.
#
#   build_bundle.sh GUFO_CHECKOUT OUT_DIR [GPU_TARGETS]
#     GUFO_CHECKOUT  loom-fun/gufo on the loom-extension branch (or one with .devops/nix/loom gpuTargets)
#     GPU_TARGETS    default "gfx1151 gfx1201" (Strix Halo and Radeon AI PRO R9700)
#
# Writes OUT_DIR/gufo-loom-<gufo rev>-<ext rev>.tar.gz and its .sha256. First run builds
# ROCm's LLVM (about 40 minutes, 24 cores); the Nix store lives in the docker
# volume `loom-bundle-nix`, so later runs take minutes (about 8 with a warm volume).
#
# On the runner:  tar xzf bundle.tar.gz -C DIR && DIR/run-gufo.sh serve llm --model PKG ...
# run-gufo.sh starts bin/gufo through the bundled loader and libraries, so the
# runner needs only a Linux x86-64 kernel with the amdgpu driver (/dev/kfd).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
ext=$(cd "$here/../.." && pwd)
gufo=$(cd "${1:?GUFO_CHECKOUT}" && pwd)
out=$(mkdir -p "${2:?OUT_DIR}" && cd "$2" && pwd)
targets=${3:-"gfx1151 gfx1201"}
volume=${NIX_VOLUME:-loom-bundle-nix}
docker volume create "$volume" >/dev/null
gufo_rev=$(git -C "$gufo" rev-parse --short HEAD); ext_rev=$(git -C "$ext" rev-parse --short HEAD)
name=gufo-loom-$gufo_rev-$ext_rev
work=$(mktemp -d "${TMPDIR:-/var/tmp}/loom-bundle.XXXXXX")   # on disk, not tmpfs
trap 'rm -rf "$work"' EXIT
docker run --rm --name "loom-bundle-$$" -v "$volume":/nix -v "$gufo":/src/gufo:ro -v "$ext":/ext:ro -v "$work":/out \
  -v "$here":/recipe:ro -e TARGETS="$targets" -e NAME="$name" -e OWNER="$(id -u):$(id -g)" nixos/nix:latest sh /recipe/in_container.sh
tar -C "$work" -czf "$out/$name.tar.gz" "$name"
(cd "$out" && sha256sum "$name.tar.gz" > "$name.tar.gz.sha256")
echo "$out/$name.tar.gz"
