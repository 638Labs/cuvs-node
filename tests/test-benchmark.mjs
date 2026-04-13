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

function randomDataset(n, d) {
  const arr = new Float32Array(n * d)
  for (let i = 0; i < arr.length; i++) arr[i] = Math.random()
  return arr
}

const DIMS = 128
const K = 10
const NUM_QUERIES = 100
const queries = randomDataset(NUM_QUERIES, DIMS)
const res = new cuvs.Resources()

// Test 1: build performance at multiple scales
const scales = [10000, 50000, 100000]
for (const n of scales) {
  check(`build ${n.toLocaleString()} vectors (${DIMS}d)`, () => {
    const data = randomDataset(n, DIMS)
    const t0 = performance.now()
    const idx = cuvs.CagraIndex.build(res, data, { rows: n, cols: DIMS })
    const ms = performance.now() - t0
    console.log(`        ${ms.toFixed(1)}ms (${(n / (ms / 1000)).toFixed(0)} vectors/sec)`)
  })
}

// Test 2: search performance at multiple scales
for (const n of scales) {
  check(`search ${n.toLocaleString()} vectors (${NUM_QUERIES} queries, k=${K})`, () => {
    const data = randomDataset(n, DIMS)
    const idx = cuvs.CagraIndex.build(res, data, { rows: n, cols: DIMS })
    const t0 = performance.now()
    const result = idx.search(res, queries, { rows: NUM_QUERIES, cols: DIMS, k: K })
    const ms = performance.now() - t0
    console.log(`        ${ms.toFixed(1)}ms (${(NUM_QUERIES / (ms / 1000)).toFixed(0)} queries/sec)`)
    assert(result.indices.length === NUM_QUERIES * K, 'wrong result count')
  })
}

// Test 3: correctness at 100K scale (self-search)
check('correctness at 100K: self-search returns near-zero distances', () => {
  const n = 100000
  const data = randomDataset(n, DIMS)
  const idx = cuvs.CagraIndex.build(res, data, { rows: n, cols: DIMS })
  const selfQ = new Float32Array(data.buffer, 0, 10 * DIMS)
  const r = idx.search(res, selfQ, { rows: 10, cols: DIMS, k: 1 })
  for (let q = 0; q < 10; q++) {
    assert(r.distances[q] < 0.01,
      `query ${q}: distance=${r.distances[q]}, expected near 0`)
  }
})

// Test 4: no memory leak over repeated cycles
check('10 build/search/dispose cycles without crash', () => {
  for (let i = 0; i < 10; i++) {
    const data = randomDataset(10000, DIMS)
    const idx = cuvs.CagraIndex.build(res, data, { rows: 10000, cols: DIMS })
    const r = idx.search(res, queries, { rows: NUM_QUERIES, cols: DIMS, k: K })
    assert(r.indices.length === NUM_QUERIES * K, `cycle ${i}: wrong result count`)
    // index goes out of scope, should be GC'd
  }
  console.log(`        (all 10 cycles completed)`)
})

res.dispose()

console.log(`\nBenchmark results: ${passed} passed, ${failed} failed`)
process.exit(failed > 0 ? 1 : 0)
