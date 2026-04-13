#!/bin/bash
set -e

if [ -z "$CONDA_PREFIX" ] && [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
    conda activate cuvs-dev
fi

echo "=========================================="
echo "  cuvs-node: verify"
echo "=========================================="

echo ""
echo "=== GPU ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi -L
else
    echo "ERROR: nvidia-smi not found"
    exit 1
fi
echo ""

echo "=== Node.js ==="
if command -v node &> /dev/null; then
    echo "  node: $(node --version)"
else
    echo "ERROR: node not found"
    exit 1
fi
echo ""

echo "=== cuVS installation ==="
if [ -z "$CONDA_PREFIX" ]; then
    echo "ERROR: CONDA_PREFIX not set. Run: conda activate cuvs-dev"
    exit 1
fi
for f in \
    "$CONDA_PREFIX/lib/libcuvs_c.so" \
    "$CONDA_PREFIX/include/cuvs/core/c_api.h" \
    "$CONDA_PREFIX/include/cuvs/neighbors/cagra.h" \
    "$CONDA_PREFIX/include/dlpack/dlpack.h"
do
    if [ -e "$f" ]; then
        echo "  FOUND: $f"
    else
        echo "  MISSING: $f"
        exit 1
    fi
done
echo ""

echo "=== Native addon ==="
ADDON="build/Release/cuvs_node.node"
if [ -e "$ADDON" ]; then
    echo "  FOUND: $ADDON"
else
    echo "  MISSING: $ADDON (run: npm run build)"
    exit 1
fi
echo ""

echo "=== Running full test suite ==="
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
node tests/test-all.mjs
