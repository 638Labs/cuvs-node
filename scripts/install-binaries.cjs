#!/usr/bin/env node
// Runs as npm postinstall.
// Downloads the prebuilt linux-x64 binary tarball from the GitHub Release
// matching this package's version, and extracts it to prebuilds/linux-x64/.
//
// Skips on platforms we don't support (binary is linux-x64-only).
// Skips if the binary is already present (re-install / cached layer).

const fs = require('fs')
const https = require('https')
const path = require('path')
const { execSync } = require('child_process')

const pkg = require('../package.json')
const VERSION = pkg.version
const PLATFORM = process.platform
const ARCH = process.arch

// We only ship a linux-x64 binary. Other platforms: bail with a clear note,
// but exit 0 so installs don't fail in CI matrixes (Mac/Windows users may be
// pulling cuvs-node transitively without ever using it).
if (PLATFORM !== 'linux' || ARCH !== 'x64') {
    console.log(`cuvs-node: no prebuilt binary for ${PLATFORM}-${ARCH}. Skipping postinstall.`)
    console.log(`           (cuVS itself only runs on Linux x86_64 with NVIDIA GPU.)`)
    process.exit(0)
}

const PREBUILDS_DIR = path.join(__dirname, '..', 'prebuilds', 'linux-x64')
const MARKER = path.join(PREBUILDS_DIR, 'libcuvs_c.so')

if (fs.existsSync(MARKER)) {
    console.log('cuvs-node: prebuilt binary already installed at ' + PREBUILDS_DIR)
    process.exit(0)
}

const URL = `https://github.com/638Labs/cuvs-node/releases/download/v${VERSION}/prebuilds-linux-x64.tar.gz`
const TARBALL = path.join(__dirname, '..', 'prebuilds-download.tar.gz')

fs.mkdirSync(PREBUILDS_DIR, { recursive: true })

console.log('cuvs-node: downloading prebuilt binary')
console.log('  url:    ' + URL)
console.log('  total:  ~1.8 GB compressed, ~2.6 GB extracted. One-time download per install.')

function download(url, dest, cb, redirectsLeft = 5) {
    https.get(url, (response) => {
        // GitHub redirects to its asset CDN — follow.
        if (response.statusCode >= 300 && response.statusCode < 400 && response.headers.location) {
            response.resume()
            if (redirectsLeft <= 0) return cb(new Error('too many redirects'))
            return download(response.headers.location, dest, cb, redirectsLeft - 1)
        }
        if (response.statusCode !== 200) {
            response.resume()
            return cb(new Error(`HTTP ${response.statusCode} fetching ${url}`))
        }
        const total = parseInt(response.headers['content-length'], 10) || 0
        let received = 0
        let lastLog = Date.now()
        const file = fs.createWriteStream(dest)
        response.on('data', (chunk) => {
            received += chunk.length
            const now = Date.now()
            if (now - lastLog > 3000 && total) {
                const pct = ((received / total) * 100).toFixed(1)
                const mb = (received / 1024 / 1024).toFixed(0)
                const totalMb = (total / 1024 / 1024).toFixed(0)
                process.stdout.write(`\r  progress: ${pct}% (${mb} / ${totalMb} MB)`)
                lastLog = now
            }
        })
        response.pipe(file)
        file.on('finish', () => {
            if (total) process.stdout.write('\n')
            file.close(() => cb(null))
        })
        file.on('error', (err) => {
            try { fs.unlinkSync(dest) } catch (_) {}
            cb(err)
        })
    }).on('error', cb)
}

download(URL, TARBALL, (err) => {
    if (err) {
        console.error('\ncuvs-node: download failed: ' + err.message)
        console.error('  You can manually download from ' + URL)
        console.error('  and extract it into ' + PREBUILDS_DIR)
        // exit 1 here would fail npm install. For now we exit 0 and let
        // require() at runtime surface a clear error if the binary is missing.
        process.exit(1)
    }
    console.log('cuvs-node: extracting...')
    try {
        execSync(`tar xzf ${TARBALL} -C ${PREBUILDS_DIR}`, { stdio: 'inherit' })
    } catch (e) {
        console.error('cuvs-node: extraction failed: ' + e.message)
        process.exit(1)
    }
    try { fs.unlinkSync(TARBALL) } catch (_) {}
    console.log('cuvs-node: installed prebuilt binary to ' + PREBUILDS_DIR)
})
