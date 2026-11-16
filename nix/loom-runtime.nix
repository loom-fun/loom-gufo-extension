# The Loom runtime this extension links. Loom's text model
# layer (the kernels and programs of a Loom package, without loom-serve's
# scheduler, caches or HTTP) behind a small C API, `libloom_text_embed.so`,
# plus loom-serve for comparison. Built from the hrx-system release that
# loom-fun packages target (hrx-loom 0.1.0).
{
  lib,
  fetchFromGitHub,
  stdenvNoCC,
  cacert,
  cmake,
  curl,
  git,
  ninja,
  python3,
  rocmPackages,
  # The embedding library's sources: a directory with `CMakeLists.txt`,
  # added to Loom's serve tree.
  embed,
  # Also build loom-serve and loom-fetch (for side-by-side comparisons);
  # the extension needs only the library.
  withTools ? false,
  # GPU targets the library carries device code for. gfx1151 is Strix Halo;
  # add gfx1201 for the Radeon AI PRO R9700 (one library serves both).
  gpuTargets ? [ "gfx1151" ],
}:

let
  # hrx-system v0.1.0, the release of hrx-loom 0.1.0.
  revision = "d01d1c18b154189fd073777329052d23f6071f14";

  src = fetchFromGitHub {
    owner = "ROCm";
    repo = "hrx-system";
    rev = revision;
    hash = "sha256-n0oHtCHMOvoQltPGkTBs2CYXdu/ulp1VF0NT8WUqiBQ=";
  };

  # Every download pinned in MODULE.cmake.lock and Loom's ISA XML descriptor,
  # fetched once so the configure step can run without network access.
  deps = stdenvNoCC.mkDerivation {
    pname = "hrx-system-deps";
    version = "0.1.0";
    inherit src;
    nativeBuildInputs = [
      cacert
      curl
      python3
    ];
    dontConfigure = true;
    buildPhase = ''
      runHook preBuild
      python3 ${./hrx-deps.py} fetch . $out
      runHook postBuild
    '';
    dontInstall = true;
    dontFixup = true;
    outputHashMode = "recursive";
    outputHashAlgo = "sha256";
    outputHash = "sha256-U24lLE2VyekJOnlJfK2EHTXqp9T0doenj8fURRTm5OI=";
  };

  llvm = rocmPackages.llvm;
  loomTools = [ "loom-serve" ];
in
llvm.rocmClangStdenv.mkDerivation {
  pname = "loom-runtime";
  version = "0.1.0";
  inherit src;

  postPatch = ''
    python3 ${./hrx-deps.py} rewrite . ${deps}
    # v0.1.0 also spells the ISA XML archive URL in CMake; use the fetched copy.
    substituteInPlace loom/src/loom/target/arch/amdgpu/descriptors/CMakeLists.txt \
      --replace-fail "https://gpuopen.com/download/AMD_GPU_MR_ISA_XML_2026_03_05.zip" \
      "file://${deps}/1f17587abc9d0d355cf6f65bdba7496838b4e8313922769199689615c434965c-AMD_GPU_MR_ISA_XML_2026_03_05.zip"
    cp -r ${embed} loom/src/loom/serve/embed
    chmod -R u+w loom/src/loom/serve/embed
  '';

  # The pinned ROCm toolchain already includes lld. Depending on the separate
  # llvm.lld derivation needlessly rebuilds LLVM instead of using that linker.
  nativeBuildInputs = [
    cmake
    ninja
    git
    python3
    llvm.clang
  ];

  cmakeFlags = [
    "-DCMAKE_BUILD_TYPE=Release"
    # Loom's static libraries link into the shared embedding library.
    "-DCMAKE_POSITION_INDEPENDENT_CODE=ON"
    "-DIREE_ROCM_DEPENDENCY_MODE=pinned"
    "-DIREE_HAL_AMDGPU_TARGETS=${lib.concatStringsSep ";" gpuTargets}"
    "-DIREE_HAL_AMDGPU_DEVICE_TOOLCHAIN=llvm-tools"
    "-DIREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_LLVM_TOOLS_DIR=${llvm.clang}/bin"
    "-DIREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_ROCM_DEVICE_LIB_PATH=${rocmPackages.rocm-device-libs}/amdgcn/bitcode"
    "-DIREE_BUILD_TESTS=OFF"
    "-DIREE_ENABLE_WERROR_FLAG=OFF"
    "-DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld"
    "-DCMAKE_SHARED_LINKER_FLAGS=-fuse-ld=lld"
    "-DCMAKE_MODULE_LINKER_FLAGS=-fuse-ld=lld"
  ]
  ++ [
    "-DLOOM_TARGET_ALL_AMD=ON"
    "-DLOOM_ARCH_AMDGPU_TARGETS=${lib.concatStringsSep ";" gpuTargets}"
    "-DLOOM_IMPORT_CXX=ON"
  ];

  ninjaFlags = [
    "loom/src/loom/serve/embed/libloom_text_embed.so"
  ]
  ++ lib.optionals withTools (
    map (tool: "loom/src/loom/tools/${tool}/${tool}") loomTools
  );

  installPhase = ''
    runHook preInstall
    install -Dm755 loom/src/loom/serve/embed/libloom_text_embed.so \
      $out/lib/libloom_text_embed.so
    install -Dm644 ${embed}/loom_text_embed.h $out/include/loom_text_embed.h
    mkdir -p $out/lib/pkgconfig
    cat > $out/lib/pkgconfig/loom-text-embed.pc <<EOF
    prefix=$out
    Name: loom-text-embed
    Description: Loom text model layer for embedding
    Version: 0.1.0
    Libs: -L$out/lib -Wl,-rpath,$out/lib -lloom_text_embed
    Cflags: -I$out/include
    EOF
  ''
  + lib.optionalString withTools ''
    for tool in ${lib.concatStringsSep " " loomTools}; do
      install -Dm755 loom/src/loom/tools/$tool/$tool $out/bin/$tool
    done
    # loom-fetch is the pure-Python model acquisition tool.
    mkdir -p $out/share/loom
    cp -r $src/loom/py $out/share/loom/py
    cat > $out/bin/loom-fetch <<EOF
    #!${python3}/bin/python3
    import sys
    sys.path.insert(0, "$out/share/loom/py")
    from loom.model.fetch_main import main
    sys.exit(main())
    EOF
    chmod +x $out/bin/loom-fetch
  ''
  + ''
    runHook postInstall
  '';

  passthru = { inherit deps revision; };

  meta = {
    description = "Loom 0.1.0 text model runtime for the Gufo loom extension (experimental)";
    homepage = "https://github.com/ROCm/hrx-system";
    license = with lib.licenses; [
      asl20
      llvm-exception
    ];
    platforms = [ "x86_64-linux" ];
  };
}
