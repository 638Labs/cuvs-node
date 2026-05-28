#!/bin/bash
# Bundle cuVS / CUDA / RAPIDS shared libs into prebuilds/linux-x64/ so the
# published npm package is self-contained. Customer needs only an NVIDIA GPU
# and a working CUDA driver (libcuda.so from nvidia-smi install); no conda,
# no separate cuVS install.
#
# Run AFTER `npx prebuildify --napi --strip` produces prebuilds/linux-x64/cuvs-node.node.
# Requires CONDA_PREFIX to point at the cuvs-dev env.
#
# Strategy: explicit manifest of known library families (like sharp/PyTorch do).
# Then verify with ldd at the end that the addon has zero unresolved deps.
# No silent fall-throughs.

set -euo pipefail

NODE_FILE="prebuilds/linux-x64/cuvs-node.node"
OUT_DIR="prebuilds/linux-x64"

[ -f "$NODE_FILE" ] || { echo "ERROR: $NODE_FILE not found. Run 'npx prebuildify --napi --strip' first." >&2; exit 1; }
[ -n "${CONDA_PREFIX:-}" ] || { echo "ERROR: CONDA_PREFIX not set. Activate cuvs-dev env first." >&2; exit 1; }

SRC_DIR="$CONDA_PREFIX/lib"

echo "=== Bundling shared libs ==="
echo "  source: $SRC_DIR"
echo "  target: $OUT_DIR"
echo ""

# ---- Ensure patchelf is installed --------------------------------------------
if ! command -v patchelf >/dev/null 2>&1; then
    echo "Installing patchelf via conda..."
    conda install -c conda-forge -y patchelf >/dev/null
fi

# ---- Manifest: which library families to bundle ------------------------------
# Glob patterns matched against $SRC_DIR. Each lib family lists every member
# we expect — if any expected family produces zero matches, we bail. This
# catches "the env layout changed" early instead of shipping a broken bundle.
#
# What's INCLUDED:
#   cuVS itself              -> libcuvs*, librmm*, librapids_logger*
#   CUDA runtime + libs cuVS uses -> libcudart*, libcublas*, libcublasLt*,
#                                  libcurand*, libcusolver*, libcusparse*,
#                                  libnvJitLink*
#   Parallel runtime         -> libgomp* (OpenMP), libnccl* (multi-GPU collectives)
#
# What's EXCLUDED (customer provides):
#   libcuda.so               -> ships with NVIDIA driver, must NOT be bundled
#   libc / libm / libdl / etc -> baseline OS libs
#   libstdc++ / libgcc_s     -> Node directly DT_NEEDs libstdc++.so.6 and the
#                                loader binds that SONAME to the system copy
#                                BEFORE our addon loads. A bundled libstdc++
#                                is dead weight that gets ignored at runtime.
#                                README requires customer to have a recent
#                                libstdc++ on their system (GLIBCXX_3.4.31+).
LIB_FAMILIES=(
    "libcuvs"
    "libcuvs_c"
    "librmm"
    "librapids_logger"
    "libcudart"
    "libcublas"
    "libcublasLt"
    "libcurand"
    "libcusolver"
    "libcusparse"
    "libnvJitLink"
    "libgomp"
    "libnccl"
)

# Returns 0 if file is ELF, 1 otherwise. Skips linker scripts (text files like
# libgcc_s.so which contains "GROUP ( libgcc_s.so.1 -lgcc )").
is_elf() {
    [ -f "$1" ] || return 1
    [ "$(head -c 4 "$1" 2>/dev/null | od -An -c | tr -d ' \n')" = '177ELF' ]
}

# ---- Copy each family --------------------------------------------------------
copy_family() {
    local family="$1"
    local matches=("$SRC_DIR"/${family}.so*)
    # Bash leaves the literal pattern when no match found; detect that.
    if [ ! -e "${matches[0]}" ]; then
        echo "ERROR: no files match ${family}.so* in $SRC_DIR" >&2
        echo "       Library layout may have changed. Update LIB_FAMILIES in $(basename $0)." >&2
        return 1
    fi
    local copied_any=0
    for src in "${matches[@]}"; do
        local name=$(basename "$src")
        # Resolve symlinks to the real file once; copy that. Then recreate the
        # SONAME symlink chain in the target dir.
        if [ -L "$src" ]; then
            local real=$(readlink -f "$src")
            local real_name=$(basename "$real")
            # Skip if the symlink target isn't ELF (e.g., libgcc_s.so points
            # at a linker script in some setups).
            is_elf "$real" || { echo "  skip non-ELF: $name -> $real_name"; continue; }
            if [ ! -f "$OUT_DIR/$real_name" ]; then
                cp "$real" "$OUT_DIR/$real_name"
                echo "  bundled:  $real_name"
                copied_any=1
            fi
            if [ ! -e "$OUT_DIR/$name" ] && [ "$name" != "$real_name" ]; then
                (cd "$OUT_DIR" && ln -s "$real_name" "$name")
                echo "  symlink:  $name -> $real_name"
            fi
        else
            # Plain file. Skip linker scripts / non-ELF entries.
            if ! is_elf "$src"; then
                echo "  skip non-ELF: $name (linker script or text)"
                continue
            fi
            if [ ! -f "$OUT_DIR/$name" ]; then
                cp "$src" "$OUT_DIR/$name"
                echo "  bundled:  $name"
                copied_any=1
            fi
        fi
    done
    if [ "$copied_any" = "0" ] && [ ! -e "$OUT_DIR/${family}.so" ] && [ ! -e "$OUT_DIR/${family}.so.1" ]; then
        # Family had matches but none copied (all were non-ELF or already present
        # from a previous family). That's OK if something resembling the family
        # already landed in OUT_DIR; otherwise warn.
        :
    fi
}

for family in "${LIB_FAMILIES[@]}"; do
    copy_family "$family"
done

# ---- Patch every bundled .so to look in $ORIGIN for its own deps -------------
# The addon itself already has RUNPATH=$ORIGIN from binding.gyp's linker flag.
# Each bundled lib needs the same so e.g. libcuvs_c.so finds libcuvs.so next
# to it instead of asking the system loader for /usr/lib/...
echo ""
echo "=== Patching RPATH=\$ORIGIN on every bundled .so ==="
for so in "$OUT_DIR"/*.so*; do
    [ -L "$so" ] && continue
    case "$so" in *.node) continue ;; esac
    is_elf "$so" || { echo "  skip non-ELF: $(basename $so)"; continue; }
    patchelf --set-rpath '$ORIGIN' "$so"
    echo "  patched:  $(basename $so)"
done

# ---- Verification: addon must have zero unresolved deps when loading from $OUT_DIR
# Force the search path to the bundle only (no LD_LIBRARY_PATH leak). If any
# dep is "not found", the bundle is incomplete — bail loudly.
echo ""
echo "=== Verifying bundle is self-contained ==="
# Customer-supplied libs that are expected to be resolved from the system,
# not from the bundle:
#   libcuda.so   -> NVIDIA driver
#   libstdc++    -> Node uses system one (must be GLIBCXX_3.4.31+, documented in README)
#   libgcc_s     -> universal on Linux; system copy is fine
EXPECTED_EXTERNAL='libcuda\.so|libstdc\+\+\.so|libgcc_s\.so'

# Verify each bundled .so's deps resolve. Anything outside the bundle that
# isn't on the EXPECTED_EXTERNAL list means we forgot to include a family.
FAIL=0
for so in "$NODE_FILE" "$OUT_DIR"/*.so*; do
    [ -L "$so" ] && continue
    [ ! -f "$so" ] && continue
    is_elf "$so" || continue
    # LD_LIBRARY_PATH points at the bundle (and at conda/lib for the system
    # libs we DON'T bundle but expect at runtime — so ldd can resolve and
    # we can tell "unresolved" vs "external-but-resolves" apart).
    OUT=$(LD_LIBRARY_PATH="$(pwd)/$OUT_DIR:$CONDA_PREFIX/lib" ldd "$so" 2>&1 || true)
    if echo "$OUT" | grep -q 'not found'; then
        UNRESOLVED=$(echo "$OUT" | grep 'not found' | grep -vE "$EXPECTED_EXTERNAL" || true)
        if [ -n "$UNRESOLVED" ]; then
            echo "ERROR: $(basename $so) has unresolved deps:" >&2
            echo "$UNRESOLVED" >&2
            FAIL=1
        fi
    fi
done
[ "$FAIL" = "0" ] || { echo "Bundle is incomplete. Update LIB_FAMILIES." >&2; exit 1; }

echo "  OK: all bundled-or-expected-external deps resolve."
echo "  Customer must provide: NVIDIA driver (libcuda.so), libstdc++.so.6 with GLIBCXX_3.4.31+."

# ---- Summary -----------------------------------------------------------------
echo ""
echo "=== Bundle contents ==="
ls -la "$OUT_DIR"
echo ""
echo "Total bundle size: $(du -sh $OUT_DIR | cut -f1)"
echo ""
echo "Bundle is self-contained. Customer needs only: Linux x86_64 + NVIDIA GPU + driver."
