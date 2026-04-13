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

const NUM_VECTORS = 10000
const DIMS = 128
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

const index = cuvs.CagraIndex.build(res, dataset, { rows: NUM_VECTORS, cols: DIMS })
const result = index.search(res, queries, { rows: NUM_QUERIES, cols: DIMS, k: K })

// Test 1: return shape
check('search returns { indices, distances }', () => {
  assert(result, 'result is null')
  assert(result.indices, 'result.indices is missing')
  assert(result.distances, 'result.distances is missing')
})

// Test 2: indices length
check('indices is Uint32Array of correct length', () => {
  assert(result.indices instanceof Uint32Array, `indices is ${result.indices.constructor.name}, expected Uint32Array`)
  assert(result.indices.length === NUM_QUERIES * K, `indices length ${result.indices.length}, expected ${NUM_QUERIES * K}`)
})

// Test 3: distances length
check('distances is Float32Array of correct length', () => {
  assert(result.distances instanceof Float32Array, `distances is ${result.distances.constructor.name}, expected Float32Array`)
  assert(result.distances.length === NUM_QUERIES * K, `distances length ${result.distances.length}, expected ${NUM_QUERIES * K}`)
})

// Test 4: valid index range
check('all index values in valid range [0, numVectors)', () => {
  for (let i = 0; i < result.indices.length; i++) {
    assert(result.indices[i] >= 0 && result.indices[i] < NUM_VECTORS,
      `index[${i}] = ${result.indices[i]} out of range`)
  }
})

// Test 5: non-negative distances
check('all distances are non-negative', () => {
  for (let i = 0; i < result.distances.length; i++) {
    assert(result.distances[i] >= 0, `distance[${i}] = ${result.distances[i]} is negative`)
  }
})

// Test 6: distances sorted per query
check('distances sorted ascending per query', () => {
  for (let q = 0; q < NUM_QUERIES; q++) {
    for (let i = 1; i < K; i++) {
      const prev = result.distances[q * K + i - 1]
      const curr = result.distances[q * K + i]
      assert(curr >= prev - 1e-6,
        `query ${q}: distance[${i-1}]=${prev} > distance[${i}]=${curr}`)
    }
  }
})

// Test 7: search performance
check('search completes under 1s', () => {
  const t0 = performance.now()
  index.search(res, queries, { rows: NUM_QUERIES, cols: DIMS, k: K })
  const elapsed = performance.now() - t0
  console.log(`        (search took ${elapsed.toFixed(1)}ms)`)
  assert(elapsed < 1000, `search took ${elapsed.toFixed(0)}ms, exceeds 1s`)
})

// Test 8: k=1
check('k=1 returns 1 result per query', () => {
  const r = index.search(res, queries, { rows: NUM_QUERIES, cols: DIMS, k: 1 })
  assert(r.indices.length === NUM_QUERIES, `expected ${NUM_QUERIES} indices, got ${r.indices.length}`)
  assert(r.distances.length === NUM_QUERIES, `expected ${NUM_QUERIES} distances, got ${r.distances.length}`)
})

// Test 9: self-search (query dataset against itself, first result should be ~0 distance)
check('self-search: first neighbor distance is near zero', () => {
  // Use first 5 vectors from dataset as queries
  const selfQueries = new Float32Array(dataset.buffer, 0, 5 * DIMS)
  const r = index.search(res, selfQueries, { rows: 5, cols: DIMS, k: 1 })
  for (let q = 0; q < 5; q++) {
    assert(r.distances[q] < 0.01,
      `self-query ${q}: distance=${r.distances[q]}, expected near 0`)
  }
})

res.dispose()

console.log(`\nCAGRA search results: ${passed} passed, ${failed} failed`)
process.exit(failed > 0 ? 1 : 0)
