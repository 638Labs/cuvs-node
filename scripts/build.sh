#!/bin/bash
set -e

if [ -z "$CONDA_PREFIX" ]; then
    echo "ERROR: conda environment not active."
    echo "Run: conda activate cuvs-dev"
    exit 1
fi

echo "Building cuvs-node..."
echo "  CONDA_PREFIX: $CONDA_PREFIX"
echo "  libcuvs_c.so: $(ls $CONDA_PREFIX/lib/libcuvs_c.so 2>/dev/null || echo 'NOT FOUND')"
echo ""

npm run build

echo ""
echo "Build complete: build/Release/cuvs_node.node"
ls -la build/Release/cuvs_node.node
