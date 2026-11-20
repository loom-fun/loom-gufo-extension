{
  description = "Gufo with the Loom extension: loom-fun models on AMD Loom kernels";

  # The Gufo this extension is tested with (flake.lock pins the revision;
  # gufo.cachix.org serves its binaries).
  inputs.gufo.url = "github:gufo-org/gufo";

  outputs =
    { self, gufo }:
    let
      system = "x86_64-linux";
      pkgs = import gufo.inputs.nixpkgs { inherit system; };
      base = gufo.packages.${system}.default;
      # libloom_text_embed: Loom's text model layer, from the pinned
      # hrx-system release, built with the nixpkgs Gufo pins.
      runtime = import ./nix/runtime.nix { inherit gufo system; };
      # This extension built as a module for `base`, with the same
      # `gufo extension install` users run (Gufo's withExtensions, plus the
      # Loom library on CMAKE_PREFIX_PATH).
      modules = pkgs.runCommand "gufo-loom-module-${base.version}" { } ''
        export HOME=$TMPDIR
        export GUFO_SDK_DIR=${base.sdk}/lib/gufo/sdk
        export GUFO_EXTENSIONS_DIR=$out
        export CMAKE_PREFIX_PATH=${runtime}
        ${base}/bin/gufo extension install ${self} --yes
        rm -f $out/.lock
        ${base}/bin/gufo extension list
      '';
      # `gufo` loading the module; `gufo extension install` still works.
      default =
        pkgs.runCommand "gufo-loom-${base.version}"
          {
            nativeBuildInputs = [ pkgs.makeWrapper ];
            passthru = { inherit modules runtime; };
            meta.mainProgram = "gufo";
          }
          ''
            mkdir -p $out/bin
            makeWrapper ${base}/bin/gufo $out/bin/gufo \
              --set-default GUFO_EXTENSIONS_DIR ${modules} \
              --set-default GUFO_SDK_DIR ${base.sdk}/lib/gufo/sdk
            ln -s gufo $out/bin/gufo-server
          '';
    in
    {
      packages.${system} = {
        inherit default runtime;
      };
      apps.${system}.default = {
        type = "app";
        program = "${default}/bin/gufo";
      };
    };
}
