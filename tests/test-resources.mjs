import { createRequire } from 'module'
const require = createRequire(import.meta.url)

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

// Test 1: addon loads
let cuvs
check('native addon loads', () => {
  cuvs = require('../build/Release/cuvs_node.node')
  assert(cuvs, 'module is null')
  assert(typeof cuvs.Resources === 'function', 'Resources class not found')
})

// Test 2: create resources
let res
check('new Resources() succeeds', () => {
  res = new cuvs.Resources()
  assert(res, 'resources is null')
})

// Test 3: dispose
check('resources.dispose() succeeds', () => {
  res.dispose()
})

// Test 4: double dispose (idempotent)
check('double dispose does not crash', () => {
  res.dispose()
})

// Test 5: multiple instances
check('multiple Resources instances', () => {
  const a = new cuvs.Resources()
  const b = new cuvs.Resources()
  a.dispose()
  b.dispose()
})

console.log(`\nResources results: ${passed} passed, ${failed} failed`)
process.exit(failed > 0 ? 1 : 0)
