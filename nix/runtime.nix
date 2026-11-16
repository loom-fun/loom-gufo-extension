# The Loom runtime this extension links (libloom_text_embed, nix/loom-runtime.nix), built with Gufo's ROCm toolchain:
#   nix build --impure --expr "import ./nix/runtime.nix { gufo = builtins.getFlake \"path:$PWD/../gufo\"; }"
# gpuTargets: gfx1151 is Strix Halo, add gfx1201 for the Radeon AI PRO R9700. withTools adds loom-serve and loom-fetch.
{ gufo, system ? "x86_64-linux", gpuTargets ? [ "gfx1151" ], withTools ? false }:
let
  rocm = (import gufo.inputs.nixpkgs-rocm10 { inherit system; }).rocmPackages_10;
  pkgs = import gufo.inputs.nixpkgs { inherit system; overlays = [ (_: _: { rocmPackages = rocm; }) ]; };
in
pkgs.callPackage ./loom-runtime.nix {
  rocmPackages = pkgs.rocmPackages.gfx1151;
  embed = ../loom-embed;
  inherit gpuTargets withTools;
}
