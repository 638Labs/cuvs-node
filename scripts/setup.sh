#!/bin/bash
set -e

echo "============================================"
echo "  cuvs-node: setup"
echo "============================================"
echo ""

# -------------------------------------------------
# Step 1: GPU
# -------------------------------------------------
echo "=== GPU ==="
if ! command -v nvidia-smi &> /dev/null; then
    echo "FATAL: nvidia-smi not found. This machine has no NVIDIA GPU drivers."
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# -------------------------------------------------
# Step 2: CUDA headers
# -------------------------------------------------
echo "=== CUDA ==="
CUDA_INC=$(find /usr/local/cuda* -name 'cuda_runtime.h' -printf '%h' -quit 2>/dev/null || echo "")
if [ -z "$CUDA_INC" ] && [ -n "$CONDA_PREFIX" ]; then
    CUDA_INC=$(find $CONDA_PREFIX -name 'cuda_runtime.h' -printf '%h' -quit 2>/dev/null || echo "")
fi
if [ -z "$CUDA_INC" ]; then
    echo "  cuda_runtime.h: will be resolved during build"
else
    echo "  cuda_runtime.h: $CUDA_INC"
fi
echo ""

# -------------------------------------------------
# Step 3: Node.js
# -------------------------------------------------
echo "=== Node.js ==="
if command -v node &> /dev/null; then
    echo "  Already installed: $(node --version)"
else
    echo "  Installing Node.js 20 LTS..."
    ARCH=$(uname -m)
    if [ "$ARCH" = "aarch64" ]; then
        NODE_ARCH="arm64"
    else
        NODE_ARCH="x64"
    fi
    curl -fsSL https://nodejs.org/dist/v20.19.0/node-v20.19.0-linux-${NODE_ARCH}.tar.xz | tar -xJ -C /usr/local --strip-components=1 2>/dev/null || \
    curl -fsSL https://nodejs.org/dist/v20.19.0/node-v20.19.0-linux-${NODE_ARCH}.tar.xz | sudo tar -xJ -C /usr/local --strip-components=1
    echo "  Installed: $(node --version)"
fi
if ! command -v node-gyp &> /dev/null; then
    npm install -g node-gyp
fi
echo ""

# -------------------------------------------------
# Step 4: Conda + cuVS
# -------------------------------------------------
echo "=== Conda + cuVS ==="
if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    echo "  Conda already installed."
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
else
    echo "  Installing Miniforge..."
    curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    bash "Miniforge3-$(uname)-$(uname -m).sh" -b -p "$HOME/miniforge3"
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
    rm -f "Miniforge3-$(uname)-$(uname -m).sh"
fi

if conda env list 2>/dev/null | grep -q cuvs-dev; then
    echo "  cuvs-dev environment exists."
    conda activate cuvs-dev
else
    echo "  Creating cuvs-dev environment..."
    conda create -n cuvs-dev -y python=3.11
    conda activate cuvs-dev
    echo "  Installing cuVS..."
    conda install -c rapidsai -c conda-forge libcuvs cuda-version=12.6 dlpack -y
fi
echo ""

# -------------------------------------------------
# Step 5: Verify cuVS files
# -------------------------------------------------
echo "=== cuVS files ==="
MISSING=false
for f in \
    "$CONDA_PREFIX/lib/libcuvs_c.so" \
    "$CONDA_PREFIX/include/cuvs/core/c_api.h" \
    "$CONDA_PREFIX/include/cuvs/neighbors/cagra.h" \
    "$CONDA_PREFIX/include/dlpack/dlpack.h"; do
    if [ -f "$f" ]; then
        echo "  FOUND: $(basename $f)"
    else
        echo "  NOT FOUND: $f"
        MISSING=true
    fi
done
if [ "$MISSING" = true ]; then
    echo ""
    echo "FATAL: Missing cuVS files. Cannot continue."
    exit 1
fi
echo ""

# -------------------------------------------------
# Step 6: Build
# -------------------------------------------------
echo "=== Build ==="
npm install --silent 2>&1 | tail -3
echo ""
npm run build 2>&1 | tail -3
if [ ! -f "build/Release/cuvs_node.node" ]; then
    echo "FATAL: Build failed. build/Release/cuvs_node.node not found."
    exit 1
fi
echo "  Addon built: $(ls -la build/Release/cuvs_node.node | awk '{print $5}') bytes"
echo ""

# -------------------------------------------------
# Step 7: Verify
# -------------------------------------------------
echo "=== Verify ==="
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
node tests/test-all.mjs

echo ""
echo "============================================"
echo "  Done. All tests passed."
echo "============================================"
