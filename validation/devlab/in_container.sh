#!/bin/sh
# Inside nixos/nix (see build_bundle.sh): loom-runtime for $TARGETS, CPU gufo with the loom extension, bundle in /out/$NAME.
set -eu
mkdir -p /build /root/.config/nix
cat > /root/.config/nix/nix.conf <<EOF
experimental-features = nix-command flakes
sandbox = false
build-users-group =
build-dir = /build
accept-flake-config = true
EOF
git config --global --add safe.directory '*'
targets=$(for t in $TARGETS; do printf '"%s" ' "$t"; done)
RT=$(nix build -L --impure --no-link --print-out-paths --expr \
  "((builtins.getFlake \"path:/src/gufo\").packages.x86_64-linux.loom-runtime.override { embed = /ext/loom-embed; gpuTargets = [ $targets ]; })")
echo "loom-runtime: $RT"
# gufo's configure needs the NPU runtime (XRT) even for a build without HIP
XRT=$(nix build -L --no-link --print-out-paths /src/gufo#xrt-amdxdna 2>/dev/null || nix build -L --no-link --print-out-paths "path:/src/gufo#xrt-amdxdna")
echo "xrt: $XRT"
nix-shell -E 'let f = builtins.getFlake "path:/src/gufo"; p = f.inputs.nixpkgs.legacyPackages.x86_64-linux; in p.mkShell { packages = with p; [ gcc cmake ninja pkg-config icu curl libpng libjpeg libwebp libopus openssl ffmpeg-headless libuuid libuuid.dev cargo rustc git patchelf coreutils findutils ]; }'  --run "
set -eu
export PKG_CONFIG_PATH_FOR_TARGET=$RT/lib/pkgconfig:\${PKG_CONFIG_PATH_FOR_TARGET:-} PKG_CONFIG_PATH=$RT/lib/pkgconfig:\${PKG_CONFIG_PATH:-}   # the nix pkg-config wrapper reads either spelling
cmake /src/gufo -B /build/gufo -GNinja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DENGINE_ENABLE_HIP=OFF \
  -DXRT_DIR=$XRT/opt/xilinx/xrt/share/cmake/XRT -DGUFO_EXTENSIONS=loom -DGUFO_EXTENSION_DIRS=/ext
cmake --build /build/gufo --target gufo --parallel 24
dest=/out/$NAME; mkdir -p \$dest/bin \$dest/lib \$dest/validation
cp /build/gufo/gufo \$dest/bin/gufo.real
cp -r /ext/validation/. \$dest/validation/
# every shared library the binary and the embed library need, flattened beside the loader
for f in \$dest/bin/gufo.real $RT/lib/libloom_text_embed.so; do
  ldd \$f | awk '/=> \\// {print \$3} /^[[:space:]]*\\/.*ld-linux/ {print \$1}'
done | sort -u | while read -r lib; do cp -L \"\$lib\" \$dest/lib/; done
cp -L $RT/lib/libloom_text_embed.so \$dest/lib/
gcc -O1 -o \$dest/bin/embed_probe.real /ext/validation/devlab/embed_probe.c -I/ext/loom-embed -L$RT/lib -lloom_text_embed -lm
chmod -R u+w \$dest
loader=\$(ls \$dest/lib/ld-linux-x86-64.so.2)
cat > \$dest/run.sh <<'EOS'
#!/bin/sh
# run.sh PROGRAM ARGS...: bin/PROGRAM.real through the bundled loader and libraries (nothing from the host but the kernel driver).
here=\$(cd \"\$(dirname \"\$0\")\" && pwd)
prog=\$1; shift
exec \"\$here/lib/ld-linux-x86-64.so.2\" --library-path \"\$here/lib\" \"\$here/bin/\$prog.real\" \"\$@\"
EOS
printf '#!/bin/sh\nexec \"\$(dirname \"\$0\")/run.sh\" gufo \"\$@\"\n' > \$dest/run-gufo.sh
printf '#!/bin/sh\nexec \"\$(dirname \"\$0\")/run.sh\" embed_probe \"\$@\"\n' > \$dest/run-embed-probe.sh
chmod +x \$dest/run.sh \$dest/run-gufo.sh \$dest/run-embed-probe.sh
\$dest/run-gufo.sh --version | head -3
"
chown -R "$OWNER" /out
