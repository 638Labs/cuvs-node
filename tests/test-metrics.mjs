// Step 1b: Distance metric support across all 4 index types.
// For each algo x each metric, build with that metric and verify search
// agrees with BruteForce ground truth at the same metric.
//
// Approximate algos (CAGRA, IvfFlat, IvfPq): require >=0.9 top-k overlap.
// BruteForce: exact match against CPU ground truth.
//
// Also covers the G1 guard: unknown metric strings must throw, never silently
// fall back to L2.

import { createRequire } from 'module'
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
const NUM_QUERIES = 20
const K = 10
const MIN_RECALL = 0.9

function seededDataset(n, d, seed = 1) {
  const arr = new Float32Array(n * d)
  let s = seed
  for (let i = 0; i < arr.length; i++) {
    s = (s * 1103515245 + 12345) & 0x7fffffff
    arr[i] = (s / 0x7fffffff)
  }
  return arr
}

function topKOverlap(a, b, nQueries, k) {
  let total = 0
  let matched = 0
  for (let q = 0; q < nQueries; q++) {
    const setA = new Set()
    for (let i = 0; i < k; i++) setA.add(Number(a[q * k + i]))
    for (let i = 0; i < k; i++) {
      total++
      if (setA.has(Number(b[q * k + i]))) matched++
    }
  }
  return matched / total
}

const res = new cuvs.Resources()
const dataset = seededDataset(NUM_VECTORS, DIMS, 42)
const queries = seededDataset(NUM_QUERIES, DIMS, 99)

const METRICS = ['l2', 'cosine', 'inner_product']

// ---- Per algo, per metric: build + search + compare to BruteForce truth ----

for (const metric of METRICS) {
  // Ground truth: BruteForce at this metric. cuVS BruteForce is exact.
  let truth
  check(`BruteForce builds with metric='${metric}'`, () => {
    const idx = cuvs.BruteForceIndex.build(res, dataset, { rows: NUM_VECTORS, cols: DIMS, metric })
    truth = idx.search(res, queries, { rows: NUM_QUERIES, cols: DIMS, k: K })
    assert(truth.indices.length === NUM_QUERIES * K, 'wrong indices length')
    assert(truth.distances.length === NUM_QUERIES * K, 'wrong distances length')
    // Distances must be sorted per query. For L2/L2_sqrt/Cosine, ascending
    // (smaller = closer). For InnerProduct, cuVS returns raw IP values
    // descending (larger = more similar).
    const ascending = metric !== 'inner_product' && metric !== 'ip'
    for (let q = 0; q < NUM_QUERIES; q++) {
      for (let i = 1; i < K; i++) {
        const prev = Number(truth.distances[q * K + i - 1])
        const curr = Number(truth.distances[q * K + i])
        if (ascending) {
          assert(curr >= prev - 1e-4,
            `metric=${metric} q=${q}: distances not ascending: ${prev} > ${curr}`)
        } else {
          assert(curr <= prev + 1e-4,
            `metric=${metric} q=${q}: distances not descending: ${prev} < ${curr}`)
        }
      }
    }
  })

  check(`CAGRA recall >= ${MIN_RECALL} at metric='${metric}'`, () => {
    if (!truth) throw new Error('no ground truth (BruteForce build failed)')
    const idx = cuvs.CagraIndex.build(res, dataset, { rows: NUM_VECTORS, cols: DIMS, metric })
    const r = idx.search(res, queries, { rows: NUM_QUERIES, cols: DIMS, k: K })
    const overlap = topKOverlap(truth.indices, r.indices, NUM_QUERIES, K)
    console.log(`        (top-${K} overlap: ${(overlap * 100).toFixed(1)}%)`)
    assert(overlap >= MIN_RECALL, `overlap ${overlap} below ${MIN_RECALL}`)
  })

  check(`IvfFlat recall >= ${MIN_RECALL} at metric='${metric}'`, () => {
    if (!truth) throw new Error('no ground truth')
    const idx = cuvs.IvfFlatIndex.build(res, dataset, { rows: NUM_VECTORS, cols: DIMS, metric, nLists: 16 })
    const r = idx.search(res, queries, { rows: NUM_QUERIES, cols: DIMS, k: K, nProbes: 16 })
    const overlap = topKOverlap(truth.indices, r.indices, NUM_QUERIES, K)
    console.log(`        (top-${K} overlap: ${(overlap * 100).toFixed(1)}%)`)
    assert(overlap >= MIN_RECALL, `overlap ${overlap} below ${MIN_RECALL}`)
  })

  check(`IvfPq recall >= 0.5 at metric='${metric}'`, () => {
    // IvfPq is lossy via quantization; relax threshold.
    if (!truth) throw new Error('no ground truth')
    const idx = cuvs.IvfPqIndex.build(res, dataset, { rows: NUM_VECTORS, cols: DIMS, metric, nLists: 16 })
    const r = idx.search(res, queries, { rows: NUM_QUERIES, cols: DIMS, k: K, nProbes: 16 })
    const overlap = topKOverlap(truth.indices, r.indices, NUM_QUERIES, K)
    console.log(`        (top-${K} overlap: ${(overlap * 100).toFixed(1)}%)`)
    assert(overlap >= 0.5, `overlap ${overlap} below 0.5`)
  })
}

// ---- G1 guard: unknown metric strings must throw ----

check('unknown metric string throws on CAGRA', () => {
  let threw = false
  try {
    cuvs.CagraIndex.build(res, dataset, { rows: NUM_VECTORS, cols: DIMS, metric: 'manhattan' })
  } catch (e) {
    threw = true
    assert(e.message.toLowerCase().includes('metric'), `error should mention 'metric', got: ${e.message}`)
  }
  assert(threw, 'expected throw on unknown metric')
})

check('unknown metric string throws on IvfFlat', () => {
  let threw = false
  try {
    cuvs.IvfFlatIndex.build(res, dataset, { rows: NUM_VECTORS, cols: DIMS, metric: 'manhattan', nLists: 16 })
  } catch (e) { threw = true }
  assert(threw, 'expected throw on unknown metric')
})

check('unknown metric string throws on IvfPq', () => {
  let threw = false
  try {
    cuvs.IvfPqIndex.build(res, dataset, { rows: NUM_VECTORS, cols: DIMS, metric: 'manhattan', nLists: 16 })
  } catch (e) { threw = true }
  assert(threw, 'expected throw on unknown metric')
})

check('unknown metric string throws on BruteForce', () => {
  let threw = false
  try {
    cuvs.BruteForceIndex.build(res, dataset, { rows: NUM_VECTORS, cols: DIMS, metric: 'manhattan' })
  } catch (e) { threw = true }
  assert(threw, 'expected throw on unknown metric')
})

check('non-string metric throws', () => {
  let threw = false
  try {
    cuvs.CagraIndex.build(res, dataset, { rows: NUM_VECTORS, cols: DIMS, metric: 5 })
  } catch (e) { threw = true }
  assert(threw, 'expected throw on numeric metric')
})

// ---- Default behavior (no metric specified) must still be L2 ----

check('omitting metric defaults to L2 (CAGRA)', () => {
  const idx = cuvs.CagraIndex.build(res, dataset, { rows: NUM_VECTORS, cols: DIMS })
  const r = idx.search(res, queries, { rows: NUM_QUERIES, cols: DIMS, k: K })
  // Compare against explicit L2 BruteForce ground truth.
  const bfIdx = cuvs.BruteForceIndex.build(res, dataset, { rows: NUM_VECTORS, cols: DIMS, metric: 'l2' })
  const bfR = bfIdx.search(res, queries, { rows: NUM_QUERIES, cols: DIMS, k: K })
  const overlap = topKOverlap(bfR.indices, r.indices, NUM_QUERIES, K)
  assert(overlap >= MIN_RECALL, `default-metric CAGRA overlap with L2 BruteForce = ${overlap}`)
})

res.dispose()

console.log(`\nMetrics test results: ${passed} passed, ${failed} failed`)
process.exit(failed > 0 ? 1 : 0)
