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

const NUM_VECTORS = 2000
const DIMS = 64
const NUM_QUERIES = 10
const K = 10

function randomDataset(n, d, seed = 1) {
  // Simple deterministic LCG so ground-truth and gpu see same data.
  const arr = new Float32Array(n * d)
  let s = seed
  for (let i = 0; i < arr.length; i++) {
    s = (s * 1103515245 + 12345) & 0x7fffffff
    arr[i] = (s / 0x7fffffff)
  }
  return arr
}

// Ground truth brute force on CPU (squared L2).
function cpuTopK(dataset, nVecs, dims, queries, nQueries, k) {
  const indices = new Int32Array(nQueries * k)
  const distances = new Float32Array(nQueries * k)
  const dists = new Float32Array(nVecs)
  for (let q = 0; q < nQueries; q++) {
    for (let i = 0; i < nVecs; i++) {
      let s = 0
      for (let d = 0; d < dims; d++) {
        const diff = queries[q * dims + d] - dataset[i * dims + d]
        s += diff * diff
      }
      dists[i] = s
    }
    // Build top-k via simple sort of (dist, idx) pairs.
    const pairs = []
    for (let i = 0; i < nVecs; i++) pairs.push([dists[i], i])
    pairs.sort((a, b) => a[0] - b[0])
    for (let j = 0; j < k; j++) {
      distances[q * k + j] = pairs[j][0]
      indices[q * k + j] = pairs[j][1]
    }
  }
  return { indices, distances }
}

const res = new cuvs.Resources()
const dataset = randomDataset(NUM_VECTORS, DIMS, 42)
const queries = randomDataset(NUM_QUERIES, DIMS, 7)

// ---- Build ----
let index
check('build returns BruteForceIndex instance', () => {
  index = cuvs.BruteForceIndex.build(res, dataset, { rows: NUM_VECTORS, cols: DIMS })
  assert(index, 'index is null')
})

check('build completes under 10s', () => {
  const t0 = performance.now()
  cuvs.BruteForceIndex.build(res, dataset, { rows: NUM_VECTORS, cols: DIMS })
  const elapsed = performance.now() - t0
  console.log(`        (build took ${elapsed.toFixed(1)}ms)`)
  assert(elapsed < 10000)
})

check('build rejects empty array', () => {
  let threw = false
  try {
    cuvs.BruteForceIndex.build(res, new Float32Array(0), { rows: 0, cols: DIMS })
  } catch (e) { threw = true }
  assert(threw)
})

check('build rejects mismatched rows*cols', () => {
  let threw = false
  try {
    cuvs.BruteForceIndex.build(res, dataset, { rows: NUM_VECTORS + 1, cols: DIMS })
  } catch (e) { threw = true }
  assert(threw)
})

// ---- Search ----
const result = index.search(res, queries, { rows: NUM_QUERIES, cols: DIMS, k: K })
const truth = cpuTopK(dataset, NUM_VECTORS, DIMS, queries, NUM_QUERIES, K)

check('search returns { indices, distances }', () => {
  assert(result && result.indices && result.distances)
})

check('indices is BigInt64Array of correct length', () => {
  assert(result.indices instanceof BigInt64Array)
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

// Exact correctness: brute force must match CPU ground truth exactly
// (modulo ties and floating-point differences in squared distance).
check('indices match CPU ground truth exactly', () => {
  for (let q = 0; q < NUM_QUERIES; q++) {
    for (let j = 0; j < K; j++) {
      const got = Number(result.indices[q * K + j])
      const expected = truth.indices[q * K + j]
      if (got !== expected) {
        // Allow tie-breaks: confirm distances match.
        const gotDist = result.distances[q * K + j]
        const expDist = truth.distances[q * K + j]
        assert(Math.abs(gotDist - expDist) < 1e-3,
          `q${q} j${j}: idx ${got} vs ${expected}, dist ${gotDist} vs ${expDist}`)
      }
    }
  }
})

check('distances match CPU ground truth (within tolerance)', () => {
  for (let i = 0; i < result.distances.length; i++) {
    const got = result.distances[i]
    const exp = truth.distances[i]
    assert(Math.abs(got - exp) < 1e-3,
      `distance[${i}] = ${got}, expected ${exp}`)
  }
})

check('search completes under 1s', () => {
  const t0 = performance.now()
  index.search(res, queries, { rows: NUM_QUERIES, cols: DIMS, k: K })
  const elapsed = performance.now() - t0
  console.log(`        (search took ${elapsed.toFixed(1)}ms)`)
  assert(elapsed < 1000)
})

check('k=1 returns 1 result per query', () => {
  const r = index.search(res, queries, { rows: NUM_QUERIES, cols: DIMS, k: 1 })
  assert(r.indices.length === NUM_QUERIES)
})

check('self-search: first neighbor is self with ~0 distance', () => {
  const NS = 5
  const selfQueries = new Float32Array(dataset.buffer, 0, NS * DIMS)
  const r = index.search(res, selfQueries, { rows: NS, cols: DIMS, k: 1 })
  for (let q = 0; q < NS; q++) {
    assert(Number(r.indices[q]) === q, `self-query ${q}: got idx ${r.indices[q]}`)
    assert(r.distances[q] < 1e-3, `self-query ${q}: dist ${r.distances[q]} not near 0`)
  }
})

// ---- Serialize / Deserialize round-trip ----
const tmp = mkdtempSync(join(tmpdir(), 'cuvs-brute-force-'))
const path = join(tmp, 'index.bin')

check('serialize writes index to disk', () => {
  index.serialize(res, path)
})

let loaded
check('deserialize loads index from disk', () => {
  loaded = cuvs.BruteForceIndex.deserialize(res, path)
  assert(loaded, 'loaded index is null')
})

check('deserialized index produces identical search results', () => {
  const r1 = index.search(res, queries, { rows: NUM_QUERIES, cols: DIMS, k: K })
  const r2 = loaded.search(res, queries, { rows: NUM_QUERIES, cols: DIMS, k: K })
  assert(r1.indices.length === r2.indices.length)
  for (let i = 0; i < r1.indices.length; i++) {
    assert(r1.indices[i] === r2.indices[i],
      `indices[${i}] mismatch: ${r1.indices[i]} vs ${r2.indices[i]}`)
  }
  for (let i = 0; i < r1.distances.length; i++) {
    assert(Math.abs(r1.distances[i] - r2.distances[i]) < 1e-5,
      `distances[${i}] mismatch: ${r1.distances[i]} vs ${r2.distances[i]}`)
  }
})

rmSync(tmp, { recursive: true, force: true })
res.dispose()

console.log(`\nBrute-force results: ${passed} passed, ${failed} failed`)
process.exit(failed > 0 ? 1 : 0)
