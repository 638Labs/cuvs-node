import { createRequire } from 'module'
import { tmpdir } from 'os'
import { join } from 'path'
import { mkdtempSync, rmSync, statSync } from 'fs'
const require = createRequire(import.meta.url)
const cuvs = require('../build/Release/cuvs_node.node')

let passed = 0
let failed = 0

function check(name, fn) {
  try {
    fn()
    console.log(`  PASS: ${name}`)
    passed++
  } catch (e) {
    console.log(`  FAIL: ${name}`)
    console.log(`        ${e.message}`)
    failed++
  }
}

function assert(condition, msg) {
  if (!condition) throw new Error(msg || 'assertion failed')
}

const NUM_VECTORS = 5000
const DIMS = 64
const NUM_QUERIES = 10
const K = 10

function randomDataset(n, d) {
  const arr = new Float32Array(n * d)
  for (let i = 0; i < arr.length; i++) arr[i] = Math.random()
  return arr
}

const res = new cuvs.Resources()
const dataset = randomDataset(NUM_VECTORS, DIMS)
const queries = randomDataset(NUM_QUERIES, DIMS)

// ---- Build CAGRA ----
const cagra = cuvs.CagraIndex.build(res, dataset, { rows: NUM_VECTORS, cols: DIMS })

// ---- Convert to HNSW ----
let hnsw
check('toHnsw returns HnswIndex instance', () => {
  hnsw = cagra.toHnsw(res)
  assert(hnsw, 'hnsw is null')
})

check('toHnsw completes under 30s', () => {
  const t0 = performance.now()
  cagra.toHnsw(res)
  const elapsed = performance.now() - t0
  console.log(`        (toHnsw took ${elapsed.toFixed(1)}ms)`)
  assert(elapsed < 30000)
})

// ---- Search HNSW ----
const result = hnsw.search(res, queries, { rows: NUM_QUERIES, cols: DIMS, k: K, ef: 64 })

check('search returns { indices, distances }', () => {
  assert(result && result.indices && result.distances)
})

check('indices is BigUint64Array of correct length', () => {
  assert(result.indices instanceof BigUint64Array,
    `indices is ${result.indices.constructor.name}, expected BigUint64Array`)
  assert(result.indices.length === NUM_QUERIES * K)
})

check('distances is Float32Array of correct length', () => {
  assert(result.distances instanceof Float32Array)
  assert(result.distances.length === NUM_QUERIES * K)
})

check('all index values in valid range [0, numVectors)', () => {
  for (let i = 0; i < result.indices.length; i++) {
    const v = Number(result.indices[i])
    assert(v >= 0 && v < NUM_VECTORS, `index[${i}] = ${v} out of range`)
  }
})

check('all distances are non-negative', () => {
  for (let i = 0; i < result.distances.length; i++) {
    assert(result.distances[i] >= -1e-5, `distance[${i}] = ${result.distances[i]}`)
  }
})

check('distances sorted ascending per query', () => {
  for (let q = 0; q < NUM_QUERIES; q++) {
    for (let i = 1; i < K; i++) {
      const prev = result.distances[q * K + i - 1]
      const curr = result.distances[q * K + i]
      assert(curr >= prev - 1e-5, `q${q}: ${prev} > ${curr}`)
    }
  }
})

check('search completes under 2s', () => {
  const t0 = performance.now()
  hnsw.search(res, queries, { rows: NUM_QUERIES, cols: DIMS, k: K, ef: 64 })
  const elapsed = performance.now() - t0
  console.log(`        (search took ${elapsed.toFixed(1)}ms)`)
  assert(elapsed < 2000)
})

check('self-search: first neighbor is self with ~0 distance', () => {
  const NS = 10
  const selfQueries = new Float32Array(dataset.buffer, 0, NS * DIMS)
  const r = hnsw.search(res, selfQueries, { rows: NS, cols: DIMS, k: 1, ef: 128 })
  let hits = 0
  for (let q = 0; q < NS; q++) {
    if (Number(r.indices[q]) === q) hits++
  }
  assert(hits >= NS * 0.8,
    `self-search top-1 recall ${hits}/${NS} below 80%`)
})

// ---- Serialize / Deserialize round-trip ----
const tmp = mkdtempSync(join(tmpdir(), 'cuvs-hnsw-'))
const path = join(tmp, 'index.bin')

check('serialize writes index to disk', () => {
  hnsw.serialize(res, path)
  const size = statSync(path).size
  assert(size > 0, 'serialized file is empty')
})

let loaded
check('deserialize loads index from disk', () => {
  loaded = cuvs.HnswIndex.deserialize(res, path, { dim: DIMS })
  assert(loaded, 'loaded index is null')
})

check('deserialized index produces matching search results', () => {
  const r1 = hnsw.search(res, queries, { rows: NUM_QUERIES, cols: DIMS, k: K, ef: 64 })
  const r2 = loaded.search(res, queries, { rows: NUM_QUERIES, cols: DIMS, k: K, ef: 64 })
  assert(r1.indices.length === r2.indices.length)
  for (let i = 0; i < r1.indices.length; i++) {
    assert(r1.indices[i] === r2.indices[i],
      `indices[${i}] mismatch: ${r1.indices[i]} vs ${r2.indices[i]}`)
  }
  for (let i = 0; i < r1.distances.length; i++) {
    assert(Math.abs(r1.distances[i] - r2.distances[i]) < 1e-4,
      `distances[${i}] mismatch: ${r1.distances[i]} vs ${r2.distances[i]}`)
  }
})

rmSync(tmp, { recursive: true, force: true })
res.dispose()

console.log(`\nHNSW results: ${passed} passed, ${failed} failed`)
process.exit(failed > 0 ? 1 : 0)
