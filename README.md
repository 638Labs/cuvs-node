# cuvs-node

Node.js bindings for [NVIDIA cuVS](https://github.com/rapidsai/cuvs) - GPU-accelerated vector search and clustering.

Build, search, and serialize high-performance vector indexes directly from Node.js using NVIDIA GPUs.

## Getting Started

You need a Linux machine with an NVIDIA GPU (A100, H100, or similar) and CUDA 12.x drivers. Cloud GPU instances from RunPod, Lambda, or similar providers work.

### Step 1: Install Node.js

```bash
# If root (RunPod):
curl -fsSL https://nodejs.org/dist/v20.19.0/node-v20.19.0-linux-x64.tar.xz | tar -xJ -C /usr/local --strip-components=1
npm install -g node-gyp

# If non-root (Lambda, etc):
curl -fsSL https://nodejs.org/dist/v20.19.0/node-v20.19.0-linux-x64.tar.xz | sudo tar -xJ -C /usr/local --strip-components=1
sudo npm install -g node-gyp
```

### Step 2: Clone this repo

```bash
cd /workspace
git clone https://github.com/skunkwerks2020/cuvs-node-001.git
cd cuvs-node-001/cuvs-node
```

### Step 3: Run setup

```bash
chmod +x scripts/*.sh
./scripts/setup.sh
```

This installs conda, cuVS, builds the native addon, and runs the full test suite. Takes about 15-20 minutes on a fresh instance. On a previously configured instance, it skips what is already installed and goes straight to build and verify.

When it finishes, you should see:

```
=================================
Status: ALL TESTS PASSED
```

If you see that, everything is working.

### Step 4: Development

Setup is complete. To start working on the code, activate the environment:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate cuvs-dev
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
```

Edit code, then build and test:

```bash
npm run build
./scripts/verify.sh
```

When ready to commit, deactivate conda first (conda conflicts with git's SSH):

```bash
conda deactivate && conda deactivate
unset LD_LIBRARY_PATH
git add -A && git commit -m "your message" && git push
```

Activate the environment before running any examples (see Step 4).

## Quick Example

This example builds a GPU-accelerated nearest neighbor index from 10,000 random vectors with 128 dimensions, searches it with 3 query vectors to find the 10 closest matches for each, then saves and reloads the index from disk.

See [`examples/basic.js`](examples/basic.js) for a runnable example. Run it with:

```bash
node examples/basic.js
```

## Rebuilding and Re-verifying

After making code changes:

```bash
npm run build
./scripts/verify.sh
```

## API

### Resources

Wraps GPU handles, CUDA streams, and memory pools. Create one per process, reuse across operations, dispose when done.

- `new Resources()` - allocate cuVS resources on the default GPU.
- `resources.dispose()` - free the underlying handles. Idempotent.

### CagraIndex

GPU-accelerated approximate nearest neighbor index using the CAGRA (Cuda Anns GRAph) algorithm. Graph-based; best general-purpose ANN on GPU.

- `CagraIndex.build(resources, dataset, { rows, cols })` - build an index from a `Float32Array` of `rows * cols` values laid out row-major. Returns a `CagraIndex`.
- `index.search(resources, queries, { rows, cols, k })` - search the index with a `Float32Array` of query vectors. Returns `{ indices: Uint32Array, distances: Float32Array }`, each of length `rows * k`. Results are sorted ascending by distance per query.
- `index.serialize(resources, path)` - write the index to disk.
- `CagraIndex.deserialize(resources, path)` - load a previously serialized index. Returns a `CagraIndex`.
- `index.toHnsw(resources)` - convert this CAGRA graph to an `HnswIndex` for CPU-side search. Returns an `HnswIndex`.

### IvfFlatIndex

Inverted-file index with flat (uncompressed) lists. Fast to build, exact distances within probed lists, higher memory than IVF-PQ.

- `IvfFlatIndex.build(resources, dataset, { rows, cols })` - build an index from a `Float32Array` of `rows * cols` values laid out row-major. Returns an `IvfFlatIndex`.
- `index.search(resources, queries, { rows, cols, k })` - search the index with a `Float32Array` of query vectors. Returns `{ indices: Uint32Array, distances: Float32Array }`, each of length `rows * k`. Results are sorted ascending by distance per query.
- `index.serialize(resources, path)` - write the index to disk.
- `IvfFlatIndex.deserialize(resources, path)` - load a previously serialized index. Returns an `IvfFlatIndex`.

### IvfPqIndex

Inverted-file index with product quantization. Lower memory than IVF-Flat, approximate distances via PQ codes, good for very large datasets.

- `IvfPqIndex.build(resources, dataset, { rows, cols })` - build an index from a `Float32Array` of `rows * cols` values laid out row-major. Returns an `IvfPqIndex`.
- `index.search(resources, queries, { rows, cols, k })` - search the index with a `Float32Array` of query vectors. Returns `{ indices: Uint32Array, distances: Float32Array }`, each of length `rows * k`. Results are sorted ascending by distance per query.
- `index.serialize(resources, path)` - write the index to disk.
- `IvfPqIndex.deserialize(resources, path)` - load a previously serialized index. Returns an `IvfPqIndex`.

### BruteForceIndex

Exact nearest neighbor search via brute-force pairwise distance. Ground-truth baseline and the right choice for small datasets.

- `BruteForceIndex.build(resources, dataset, { rows, cols })` - build an index from a `Float32Array` of `rows * cols` values laid out row-major. Returns a `BruteForceIndex`.
- `index.search(resources, queries, { rows, cols, k })` - search the index with a `Float32Array` of query vectors. Returns `{ indices: Uint32Array, distances: Float32Array }`, each of length `rows * k`. Results are sorted ascending by distance per query.
- `index.serialize(resources, path)` - write the index to disk.
- `BruteForceIndex.deserialize(resources, path)` - load a previously serialized index. Returns a `BruteForceIndex`.

### HnswIndex

CPU-side HNSW graph, produced by converting a GPU-built `CagraIndex`. Build on GPU for speed, serve on CPU for cheap deployment.

- `HnswIndex` instances are created via `cagraIndex.toHnsw(resources)` or `HnswIndex.deserialize(resources, path)`; there is no standalone `build` method.
- `index.search(resources, queries, { rows, cols, k })` - search the index with a `Float32Array` of query vectors. Returns `{ indices: Uint32Array, distances: Float32Array }`, each of length `rows * k`. Results are sorted ascending by distance per query.
- `index.serialize(resources, path)` - write the index to disk.
- `HnswIndex.deserialize(resources, path)` - load a previously serialized index. Returns an `HnswIndex`.

## Important: Git and Conda

Conda modifies library paths that can break git's SSH. Always deactivate conda before git operations:

```bash
conda deactivate && conda deactivate
unset LD_LIBRARY_PATH
git add -A && git commit -m "your message" && git push
```

Then reactivate when you need to build or test again:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate cuvs-dev
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
```

## License

Apache-2.0
