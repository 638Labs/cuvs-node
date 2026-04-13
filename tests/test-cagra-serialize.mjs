import { createRequire } from 'module'
import { existsSync, statSync, unlinkSync } from 'fs'
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
const INDEX_PATH = '/tmp/cuvs_node_test_index.bin'

function randomDataset(n, d) {
  const arr = new Float32Array(n * d)
  for (let i = 0; i < arr.length; i++) arr[i] = Math.random()
  return arr
}

// Cleanup from previous runs
if (existsSync(INDEX_PATH)) unlinkSync(INDEX_PATH)

const res = new cuvs.Resources()
const dataset = randomDataset(NUM_VECTORS, DIMS)
const queries = randomDataset(NUM_QUERIES, DIMS)

const index = cuvs.CagraIndex.build(res, dataset, { rows: NUM_VECTORS, cols: DIMS })
const original = index.search(res, queries, { rows: NUM_QUERIES, cols: DIMS, k: K })

// Test 1: serialize
check('serialize does not throw', () => {
  index.serialize(res, INDEX_PATH)
})

// Test 2: file exists
check('serialized file exists with size > 0', () => {
  assert(existsSync(INDEX_PATH), 'file does not exist')
  const size = statSync(INDEX_PATH).size
  console.log(`        (file size: ${(size / 1024).toFixed(1)} KB)`)
  assert(size > 0, 'file is empty')
})

// Test 3: deserialize
let loaded
check('deserialize does not throw', () => {
  loaded = cuvs.CagraIndex.deserialize(res, INDEX_PATH)
  assert(loaded, 'deserialized index is null')
})

// Test 4: same indices
check('deserialized index returns same neighbor indices', () => {
  const result = loaded.search(res, queries, { rows: NUM_QUERIES, cols: DIMS, k: K })
  for (let i = 0; i < original.indices.length; i++) {
    assert(original.indices[i] === result.indices[i],
      `index mismatch at position ${i}: original=${original.indices[i]}, loaded=${result.indices[i]}`)
  }
})

// Test 5: same distances
check('deserialized index returns same distances (within tolerance)', () => {
  const result = loaded.search(res, queries, { rows: NUM_QUERIES, cols: DIMS, k: K })
  const tolerance = 1e-5
  for (let i = 0; i < original.distances.length; i++) {
    const diff = Math.abs(original.distances[i] - result.distances[i])
    assert(diff < tolerance,
      `distance mismatch at position ${i}: original=${original.distances[i]}, loaded=${result.distances[i]}, diff=${diff}`)
  }
})

// Test 6: serialize to invalid path
check('serialize to invalid path throws', () => {
  let threw = false
  try {
    index.serialize(res, '/nonexistent/directory/index.bin')
  } catch (e) {
    threw = true
  }
  assert(threw, 'expected error on invalid path')
})

// Test 7: deserialize from non-existent file
check('deserialize from non-existent file throws', () => {
  let threw = false
  try {
    cuvs.CagraIndex.deserialize(res, '/tmp/does_not_exist_cuvs.bin')
  } catch (e) {
    threw = true
  }
  assert(threw, 'expected error on missing file')
})

// Cleanup
if (existsSync(INDEX_PATH)) unlinkSync(INDEX_PATH)
res.dispose()

console.log(`\nSerialize results: ${passed} passed, ${failed} failed`)
process.exit(failed > 0 ? 1 : 0)
