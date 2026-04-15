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
const K = 10
const NUM_QUERIES = 32

function seededRandom(seed) {
  let s = seed >>> 0
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0
    return s / 0x100000000
  }
}

function randomDataset(n, d, seed) {
  const rand = seededRandom(seed)
  const arr = new Float32Array(n * d)
  for (let i = 0; i < arr.length; i++) arr[i] = rand()
  return arr
}

function splitIntoChunks(dataset, rows, cols, nChunks) {
  const chunks = []
  const rowsPerChunk = Math.floor(rows / nChunks)
  let rowOffset = 0
  for (let i = 0; i < nChunks; i++) {
    const chunkRows = (i === nChunks - 1) ? (rows - rowOffset) : rowsPerChunk
    const start = rowOffset * cols
    const end = start + chunkRows * cols
    chunks.push(dataset.slice(start, end))
    rowOffset += chunkRows
  }
  return chunks
}

function searchIndex(IndexClass, index, res, queries, opts) {
  return index.search(res, queries, opts)
}

const res = new cuvs.Resources()
const dataset = randomDataset(NUM_VECTORS, DIMS, 42)
const queries = randomDataset(NUM_QUERIES, DIMS, 7)

// Build reference (non-chunked) index for each type
const refCagra = cuvs.CagraIndex.build(res, dataset, { rows: NUM_VECTORS, cols: DIMS })
const refIvfFlat = cuvs.IvfFlatIndex.build(res, dataset, { rows: NUM_VECTORS, cols: DIMS, nLists: 64 })
const refIvfPq = cuvs.IvfPqIndex.build(res, dataset, { rows: NUM_VECTORS, cols: DIMS, nLists: 64 })
const refBruteForce = cuvs.BruteForceIndex.build(res, dataset, { rows: NUM_VECTORS, cols: DIMS })

const refCagraResult = refCagra.search(res, queries, { rows: NUM_QUERIES, cols: DIMS, k: K })
const refIvfFlatResult = refIvfFlat.search(res, queries, { rows: NUM_QUERIES, cols: DIMS, k: K, nProbes: 8 })
const refIvfPqResult = refIvfPq.search(res, queries, { rows: NUM_QUERIES, cols: DIMS, k: K, nProbes: 8 })
const refBruteForceResult = refBruteForce.search(res, queries, { rows: NUM_QUERIES, cols: DIMS, k: K })

function sameIndices(a, b) {
  if (a.length !== b.length) return false
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return false
  }
  return true
}

function recallVsGroundTruth(result, groundTruth, k, nQueries) {
  let matches = 0
  for (let q = 0; q < nQueries; q++) {
    const gt = new Set()
    for (let i = 0; i < k; i++) gt.add(String(groundTruth[q * k + i]))
    for (let i = 0; i < k; i++) {
      if (gt.has(String(result[q * k + i]))) matches++
    }
  }
  return matches / (nQueries * k)
}

function runChunkedTests(label, IndexClass, buildOpts, searchOpts, refResult, exact) {
  const refRecall = exact
    ? 1
    : recallVsGroundTruth(refResult.indices, refBruteForceResult.indices, K, NUM_QUERIES)

  for (const nChunks of [1, 2, 10]) {
    check(`${label}: buildChunked with ${nChunks} chunk(s) matches build`, () => {
      const chunks = splitIntoChunks(dataset, NUM_VECTORS, DIMS, nChunks)
      assert(chunks.length === nChunks, 'wrong chunk count')
      const idx = IndexClass.buildChunked(res, chunks, buildOpts)
      assert(idx, 'index is null')
      const result = idx.search(res, queries, searchOpts)
      if (exact) {
        assert(sameIndices(result.indices, refResult.indices),
          'chunked results differ from reference')
      } else {
        const recall = recallVsGroundTruth(result.indices, refBruteForceResult.indices, K, NUM_QUERIES)
        const tolerance = 0.10
        assert(recall >= refRecall - tolerance,
          `chunked recall ${recall.toFixed(3)} below reference recall ${refRecall.toFixed(3)} - ${tolerance}`)
      }
    })
  }

  check(`${label}: buildChunked rejects mismatched total rows`, () => {
    const chunks = splitIntoChunks(dataset, NUM_VECTORS, DIMS, 2)
    let threw = false
    try {
      IndexClass.buildChunked(res, chunks, { ...buildOpts, rows: NUM_VECTORS + 1 })
    } catch (e) {
      threw = true
    }
    assert(threw, 'expected error on mismatched rows')
  })

  check(`${label}: buildChunked rejects empty chunks array`, () => {
    let threw = false
    try {
      IndexClass.buildChunked(res, [], buildOpts)
    } catch (e) {
      threw = true
    }
    assert(threw, 'expected error on empty chunks')
  })
}

runChunkedTests(
  'CagraIndex',
  cuvs.CagraIndex,
  { rows: NUM_VECTORS, cols: DIMS },
  { rows: NUM_QUERIES, cols: DIMS, k: K },
  refCagraResult,
  false,
)

runChunkedTests(
  'IvfFlatIndex',
  cuvs.IvfFlatIndex,
  { rows: NUM_VECTORS, cols: DIMS, nLists: 64 },
  { rows: NUM_QUERIES, cols: DIMS, k: K, nProbes: 8 },
  refIvfFlatResult,
  false,
)

runChunkedTests(
  'IvfPqIndex',
  cuvs.IvfPqIndex,
  { rows: NUM_VECTORS, cols: DIMS, nLists: 64 },
  { rows: NUM_QUERIES, cols: DIMS, k: K, nProbes: 8 },
  refIvfPqResult,
  false,
)

runChunkedTests(
  'BruteForceIndex',
  cuvs.BruteForceIndex,
  { rows: NUM_VECTORS, cols: DIMS },
  { rows: NUM_QUERIES, cols: DIMS, k: K },
  refBruteForceResult,
  true,
)

res.dispose()

console.log(`\nChunked build results: ${passed} passed, ${failed} failed`)
process.exit(failed > 0 ? 1 : 0)
