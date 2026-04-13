#!/bin/bash
set -e

if [ -z "$CONDA_PREFIX" ]; then
    echo "ERROR: conda environment not active."
    echo "Run: conda activate cuvs-dev"
    exit 1
fi

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

TEST_FILE=${1:-tests/test-resources.mjs}
echo "Running: $TEST_FILE"
echo ""
node "$TEST_FILE"
