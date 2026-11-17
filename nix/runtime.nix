# The Loom runtime this extension links (libloom_text_embed, nix/loom-runtime.nix), built with the nixpkgs Gufo pins
# (so the library loads into Gufo's process):
#   nix build --impure --expr "import ./nix/runtime.nix { gufo = builtins.getFlake \"path:$PWD/../gufo\"; }"
# One library serves every AMD GPU Loom supports (Strix Halo, Radeon AI PRO R9700, ...). withTools adds loom-serve and
# loom-fetch.
{ gufo, system ? "x86_64-linux", withTools ? false }:
let
  pkgs = import gufo.inputs.nixpkgs { inherit system; };
in
pkgs.callPackage ./loom-runtime.nix {
  embed = ../loom-embed;
  inherit withTools;
}
