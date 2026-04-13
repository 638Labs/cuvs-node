// cuvs-node basic example
//
// Demonstrates all five algorithms in cuvs-node against the same dataset:
//   1. CAGRA           - graph-based GPU ANN
//   2. IVF-Flat        - inverted-file index with flat lists
//   3. IVF-PQ          - inverted-file index with product quantization
//   4. Brute-force     - exact nearest neighbor (ground truth)
//   5. CAGRA -> HNSW   - convert GPU CAGRA graph to CPU HNSW for serving
//
// Run: node examples/basic.js

const {
  Resources,
  CagraIndex,
  IvfFlatIndex,
  IvfPqIndex,
  BruteForceIndex,
  HnswIndex,
} = require('../')

const NUM_VECTORS = 10000
const DIMS = 128
const NUM_QUERIES = 3
const K = 10

function printResults(label, indices, distances) {
  console.log(`  ${label}:`)
  for (let q = 0; q < NUM_QUERIES; q++) {
    const idx = indices.slice(q * K, q * K + 3)
    const dist = distances.slice(q * K, q * K + 3)
    console.log(`    Query ${q}: neighbors [${idx}], distances [${Array.from(dist).map(d => d.toFixed(4))}]`)
  }
}

const res = new Resources()

console.log('')
console.log('=== cuvs-node basic example ===')
console.log('')

// Shared dataset and queries
console.log(`Preparing shared dataset: ${NUM_VECTORS} random vectors x ${DIMS} dimensions...`)
const dataset = new Float32Array(NUM_VECTORS * DIMS)
for (let i = 0; i < dataset.length; i++) dataset[i] = Math.random()
const queries = new Float32Array(NUM_QUERIES * DIMS)
for (let i = 0; i < queries.length; i++) queries[i] = Math.random()
console.log(`  Dataset and ${NUM_QUERIES} query vectors generated.`)
console.log('')

// --- 1. CAGRA ---
console.log('Step 1: CAGRA (graph-based ANN, GPU)')
console.log('  Building CAGRA index on GPU...')
const cagra = CagraIndex.build(res, dataset, { rows: NUM_VECTORS, cols: DIMS })
console.log('  Built. Searching for 10 nearest neighbors...')
{
  const { indices, distances } = cagra.search(res, queries, { rows: NUM_QUERIES, cols: DIMS, k: K })
  printResults('CAGRA results', indices, distances)
}
cagra.serialize(res, './my-index.bin')
console.log('  Success - serialized to ./my-index.bin')
const cagraLoaded = CagraIndex.deserialize(res, './my-index.bin')
console.log('  Success - deserialized from ./my-index.bin')
console.log('')

// --- 2. IVF-Flat ---
console.log('Step 2: IVF-Flat (inverted-file index, uncompressed lists)')
console.log('  Building IVF-Flat index on GPU...')
const ivfFlat = IvfFlatIndex.build(res, dataset, { rows: NUM_VECTORS, cols: DIMS })
console.log('  Built. Searching for 10 nearest neighbors...')
{
  const { indices, distances } = ivfFlat.search(res, queries, { rows: NUM_QUERIES, cols: DIMS, k: K })
  printResults('IVF-Flat results', indices, distances)
}
console.log('')

// --- 3. IVF-PQ ---
console.log('Step 3: IVF-PQ (inverted-file + product quantization, low memory)')
console.log('  Building IVF-PQ index on GPU...')
const ivfPq = IvfPqIndex.build(res, dataset, { rows: NUM_VECTORS, cols: DIMS })
console.log('  Built. Searching for 10 nearest neighbors...')
{
  const { indices, distances } = ivfPq.search(res, queries, { rows: NUM_QUERIES, cols: DIMS, k: K })
  printResults('IVF-PQ results', indices, distances)
}
console.log('')

// --- 4. Brute-force ---
console.log('Step 4: Brute-force (exact nearest neighbor, ground truth)')
console.log('  Building brute-force index on GPU...')
const brute = BruteForceIndex.build(res, dataset, { rows: NUM_VECTORS, cols: DIMS })
console.log('  Built. Searching for 10 nearest neighbors...')
{
  const { indices, distances } = brute.search(res, queries, { rows: NUM_QUERIES, cols: DIMS, k: K })
  printResults('Brute-force results (exact)', indices, distances)
}
console.log('')

// --- 5. CAGRA -> HNSW ---
console.log('Step 5: CAGRA -> HNSW (convert GPU graph to CPU HNSW for serving)')
console.log('  Converting CAGRA index to HNSW on CPU...')
const hnsw = cagra.toHnsw(res)
console.log('  Converted. Searching HNSW on CPU...')
{
  const { indices, distances } = hnsw.search(res, queries, { rows: NUM_QUERIES, cols: DIMS, k: K })
  printResults('HNSW results', indices, distances)
}
console.log('')

res.dispose()
console.log('Done. GPU resources released.')
