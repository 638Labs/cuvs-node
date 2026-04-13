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

function randomDataset(n, d) {
  const arr = new Float32Array(n * d)
  for (let i = 0; i < arr.length; i++) arr[i] = Math.random()
  return arr
}

const res = new cuvs.Resources()
const dataset = randomDataset(NUM_VECTORS, DIMS)

// Test 1: build returns an instance
let index
check('build returns CagraIndex instance', () => {
  index = cuvs.CagraIndex.build(res, dataset, { rows: NUM_VECTORS, cols: DIMS })
  assert(index, 'index is null')
})

// Test 2: build does not throw
check('build does not throw on valid input', () => {
  const idx = cuvs.CagraIndex.build(res, dataset, { rows: NUM_VECTORS, cols: DIMS })
  assert(idx, 'index is null')
})

// Test 3: build performance
check('build completes under 30s', () => {
  const t0 = performance.now()
  const idx = cuvs.CagraIndex.build(res, dataset, { rows: NUM_VECTORS, cols: DIMS })
  const elapsed = performance.now() - t0
  console.log(`        (build took ${elapsed.toFixed(1)}ms)`)
  assert(elapsed < 30000, `build took ${elapsed.toFixed(0)}ms, exceeds 30s`)
})

// Test 4: reject empty array
check('build rejects empty array', () => {
  let threw = false
  try {
    cuvs.CagraIndex.build(res, new Float32Array(0), { rows: 0, cols: DIMS })
  } catch (e) {
    threw = true
  }
  assert(threw, 'expected error on empty array')
})

// Test 5: reject mismatched dimensions
check('build rejects mismatched rows*cols', () => {
  let threw = false
  try {
    cuvs.CagraIndex.build(res, dataset, { rows: NUM_VECTORS + 1, cols: DIMS })
  } catch (e) {
    threw = true
  }
  assert(threw, 'expected error on mismatched dimensions')
})

// Test 6: second build succeeds
check('second build with different data succeeds', () => {
  const dataset2 = randomDataset(5000, 64)
  const idx2 = cuvs.CagraIndex.build(res, dataset2, { rows: 5000, cols: 64 })
  assert(idx2, 'second index is null')
})

res.dispose()

console.log(`\nCAGRA build results: ${passed} passed, ${failed} failed`)
process.exit(failed > 0 ? 1 : 0)
