#!/bin/sh
# macOS only. faiss-cpu and torch wheels each bundle their own libomp.dylib; two OpenMP runtimes in one
# process abort ("OMP: Error #15") or segfault. Point faiss at torch's copy so one runtime is loaded.
# Linux (and the Docker image) are unaffected: both use the system libgomp.
set -e
SP=$(python -c "import site; print(site.getsitepackages()[0])")
FA="$SP/faiss/.dylibs/libomp.dylib"
TO="$SP/torch/lib/libomp.dylib"
[ -f "$TO" ] && [ -e "$FA" ] || { echo "nothing to do"; exit 0; }
[ -L "$FA" ] || mv "$FA" "$FA.bundled"
ln -sf "$TO" "$FA"
echo "faiss now uses torch's libomp: $(readlink "$FA")"
