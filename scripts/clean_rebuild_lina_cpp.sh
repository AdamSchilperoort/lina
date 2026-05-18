#!/usr/bin/env bash
# Nuclear rebuild of lina_cpp. Wipes every cached artifact that could
# leave a stale _core.so on PYTHONPATH, then reinstalls the editable
# package from the current source tree.
#
# Usage:
#   bash scripts/clean_rebuild_lina_cpp.sh             # CPU+CUDA auto-detect
#   LINA_USE_CUDA=1 bash scripts/clean_rebuild_lina_cpp.sh
#   LINA_USE_CUDA=0 bash scripts/clean_rebuild_lina_cpp.sh
#
# Why this script exists:
# `pip install -e .` does NOT always recompile when only C++ source files
# change. setuptools sees no new .py files and skips the build_ext step.
# The cmake build dir is also reused across runs, so a partially-built
# .so from a half-failed earlier attempt can stick around forever. The
# canonical symptom is `lina_cpp.llowfsc_reconstruct` returning a
# constant vector (every coefficient equal to coeff[0]) -- the C++
# source is correct but the compiled binary is not. Wipe everything
# below and the next install rebuilds from scratch.

set -euo pipefail

# Resolve repo root from the script location, so this works whether the
# user is sitting in repo/, repo/scripts/, or somewhere else entirely.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
cd "${REPO_ROOT}"

echo "[clean-rebuild] repo root = ${REPO_ROOT}"
echo "[clean-rebuild] python    = $(command -v python)"
echo "[clean-rebuild] cmake     = $(command -v cmake || echo 'NOT FOUND')"

echo "[clean-rebuild] removing stale build artifacts..."
rm -rf \
    lina_cpp/build \
    lina_cpp/dist \
    lina_cpp/*.egg-info \
    lina_cpp/src/lina_cpp.egg-info \
    lina_cpp/src/lina_cpp/_core*.so \
    lina_cpp/src/lina_cpp/__pycache__ \
    build/lib*/lina_cpp \
    build/temp*/lina_cpp \
    cpp/build cpp/build-cuda

echo "[clean-rebuild] uninstalling any existing lina_cpp..."
pip uninstall -y lina_cpp 2>/dev/null || true

echo "[clean-rebuild] running 'pip install -e ./lina_cpp/'..."
LINA_USE_CUDA="${LINA_USE_CUDA:-auto}" pip install -e ./lina_cpp/ \
    --no-build-isolation \
    --no-cache-dir \
    --force-reinstall

echo
echo "[clean-rebuild] running smoke test..."
python -m lina_cpp.smoke
