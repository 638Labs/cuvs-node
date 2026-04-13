import { spawnSync } from 'child_process'

const tests = [
  'tests/test-resources.mjs',
  'tests/test-cagra-build.mjs',
  'tests/test-cagra-search.mjs',
  'tests/test-cagra-serialize.mjs',
  'tests/test-ivf-flat.mjs',
  'tests/test-ivf-pq.mjs',
  'tests/test-brute-force.mjs',
  'tests/test-hnsw.mjs',
  'tests/test-benchmark.mjs',
]

let allPassed = true
let totalPassed = 0
let totalFailed = 0

console.log('=== cuvs-node: Full Test Suite ===')
console.log('')

for (const test of tests) {
  console.log(`--- ${test} ---`)
  const result = spawnSync('node', [test], { env: { ...process.env }, encoding: 'utf8' })
  if (result.stdout) process.stdout.write(result.stdout)
  if (result.stderr) process.stderr.write(result.stderr)

  const output = (result.stdout || '') + (result.stderr || '')
  const match = output.match(/(\d+)\s+passed,\s+(\d+)\s+failed/)
  if (match) {
    totalPassed += parseInt(match[1], 10)
    totalFailed += parseInt(match[2], 10)
  }

  if (result.status !== 0) {
    allPassed = false
    console.log(`\n^^^ FAILED ^^^\n`)
  } else {
    console.log('')
  }
}

console.log('=================================')
console.log(`Total: ${totalPassed} passed, ${totalFailed} failed`)
if (allPassed) {
  console.log('Status: ALL TESTS PASSED')
} else {
  console.log('Status: SOME TESTS FAILED')
}

process.exit(allPassed ? 0 : 1)
