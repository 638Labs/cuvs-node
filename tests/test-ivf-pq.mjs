import { createRequire } from 'module'
import { tmpdir } from 'os'
import { join } from 'path'
import { mkdtempSync, rmSync } from 'fs'
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

const NUM_VECTORS = 10000
const DIMS = 128
const NUM_QUERIES = 10
const K = 10
const N_LISTS = 64
const PQ_BITS = 8
const PQ_DIM = 32

function randomDataset(n, d) {
  const arr = new Float32Array(n * d)
  for (let i = 0; i < arr.length; i++) arr[i] = Math.random()
  return arr
}

const res = new cuvs.Resources()
const dataset = randomDataset(NUM_VECTORS, DIMS)
const queries = randomDataset(NUM_QUERIES, DIMS)

// ---- Build ----
let index
check('build returns IvfPqIndex instance', () => {
  index = cuvs.IvfPqIndex.build(res, dataset, {
    rows: NUM_VECTORS, cols: DIMS, nLists: N_LISTS, pqBits: PQ_BITS, pqDim: PQ_DIM,
  })
  assert(index, 'index is null')
})

check('build completes under 60s', () => {
  const t0 = performance.now()
  cuvs.IvfPqIndex.build(res, dataset, {
    rows: NUM_VECTORS, cols: DIMS, nLists: N_LISTS, pqBits: PQ_BITS, pqDim: PQ_DIM,
  })
  const elapsed = performance.now() - t0
  console.log(`        (build took ${elapsed.toFixed(1)}ms)`)
  assert(elapsed < 60000, `build took ${elapsed.toFixed(0)}ms, exceeds 60s`)
})

check('build with defaults succeeds', () => {
  const idx = cuvs.IvfPqIndex.build(res, dataset, { rows: NUM_VECTORS, cols: DIMS })
  assert(idx, 'default index is null')
})

check('build rejects empty array', () => {
  let threw = false
  try {
    cuvs.IvfPqIndex.build(res, new Float32Array(0), { rows: 0, cols: DIMS })
  } catch (e) { threw = true }
  assert(threw, 'expected error on empty array')
})

check('build rejects mismatched rows*cols', () => {
  let threw = false
  try {
    cuvs.IvfPqIndex.build(res, dataset, { rows: NUM_VECTORS + 1, cols: DIMS })
  } catch (e) { threw = true }
  assert(threw, 'expected error on mismatched dimensions')
})

// ---- Search ----
const result = index.search(res, queries, { rows: NUM_QUERIES, cols: DIMS, k: K, nProbes: 16 })

check('search returns { indices, distances }', () => {
  assert(result, 'result is null')
  assert(result.indices, 'indices missing')
  assert(result.distances, 'distances missing')
})

check('indices is BigInt64Array of correct length', () => {
  assert(result.indices instanceof BigInt64Array,
    `indices is ${result.indices.constructor.name}, expected BigInt64Array`)
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
    assert(result.distances[i] >= 0, `distance[${i}] = ${result.distances[i]} is negative`)
  }
})

check('distances sorted ascending per query', () => {
  for (let q = 0; q < NUM_QUERIES; q++) {
    for (let i = 1; i < K; i++) {
      const prev = result.distances[q * K + i - 1]
      const curr = result.distances[q * K + i]
      assert(curr >= prev - 1e-4,
        `query ${q}: distance[${i-1}]=${prev} > distance[${i}]=${curr}`)
    }
  }
})

check('search completes under 1s', () => {
  const t0 = performance.now()
  index.search(res, queries, { rows: NUM_QUERIES, cols: DIMS, k: K, nProbes: 16 })
  const elapsed = performance.now() - t0
  console.log(`        (search took ${elapsed.toFixed(1)}ms)`)
  assert(elapsed < 1000, `search took ${elapsed.toFixed(0)}ms, exceeds 1s`)
})

check('k=1 returns 1 result per query', () => {
  const r = index.search(res, queries, { rows: NUM_QUERIES, cols: DIMS, k: 1, nProbes: 16 })
  assert(r.indices.length === NUM_QUERIES)
  assert(r.distances.length === NUM_QUERIES)
})

check('self-search: top-1 recall is reasonable (PQ is lossy)', () => {
  // PQ is lossy, but with enough probes most self-queries should still return self.
  const NS = 20
  const selfQueries = new Float32Array(dataset.buffer, 0, NS * DIMS)
  const r = index.search(res, selfQueries, { rows: NS, cols: DIMS, k: 1, nProbes: N_LISTS })
  let hits = 0
  for (let q = 0; q < NS; q++) {
    if (Number(r.indices[q]) === q) hits++
  }
  assert(hits >= NS * 0.5,
    `self-search top-1 recall ${hits}/${NS} below 50%`)
})

// ---- Serialize / Deserialize round-trip ----
const tmp = mkdtempSync(join(tmpdir(), 'cuvs-ivf-pq-'))
const path = join(tmp, 'index.bin')

check('serialize writes index to disk', () => {
  index.serialize(res, path)
})

let loaded
check('deserialize loads index from disk', () => {
  loaded = cuvs.IvfPqIndex.deserialize(res, path)
  assert(loaded, 'loaded index is null')
})

check('deserialized index produces identical search results', () => {
  const r1 = index.search(res, queries, { rows: NUM_QUERIES, cols: DIMS, k: K, nProbes: 16 })
  const r2 = loaded.search(res, queries, { rows: NUM_QUERIES, cols: DIMS, k: K, nProbes: 16 })
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

console.log(`\nIVF-PQ results: ${passed} passed, ${failed} failed`)
process.exit(failed > 0 ? 1 : 0)
