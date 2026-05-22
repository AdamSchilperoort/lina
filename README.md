# Lina
Wavefront sensing and control algorithms and tools being developed in UASAL.

This repository now supports two installable Python packages:

- `lina` (pure Python reference implementation)
- `lina_cpp` (Python-compatible frontend backed by C++ kernels)

The core goal is direct apples-to-apples comparison: same input -> same output,
with performance differences measured separately.

## Install

From the repository root:

```bash
# Pure Python package
pip install -e .

# C++-backed mirror package
pip install -e ./lina_cpp
```

If you want GPU support for `lina_cpp`, install with CUDA enabled:

```bash
LINA_USE_CUDA=1 pip install -e ./lina_cpp
```

## Switch between implementations

Use either package explicitly:

```python
import lina
import lina_cpp
```

Or switch with a one-line alias:

```python
# Pure Python
import lina as wf

# C++-backed
import lina_cpp as wf
```

Both aim to expose the same public API shape (`utils`, `props`, `dm`, `efc`,
`iefc`, `aefc`, `llowfsc`, etc.), so user code can swap implementations with
minimal/no edits.

## Validate parity

Canonical numerical parity suite:

```bash
python -m pytest lina/tests/test_per_method_parity.py -q
```

Performance suite (CPU/GPU comparisons):

```bash
LINA_RUN_BENCHMARKS=1 python -m pytest lina/tests/test_benchmarks.py -q
```

## Thread tuning (CPU math)

`lina_cpp` now exposes runtime CPU thread controls:

```python
import lina_cpp
lina_cpp.set_num_threads(4)
print(lina_cpp.get_num_threads())
```

You can sweep thread counts on your host with:

```bash
python scripts/benchmark_thread_sweep.py --threads 1,2,4,8,12,16 --iters 8
```

For math/optics benchmark scripts, run capped dimensions by default and enable
full grids with:

```bash
python scripts/benchmark_basic_math_func.py --full-compute
python scripts/benchmark_optical_algorithm_bench.py --full-compute
```

To drive both scripts from one explicit compute grid file:

```bash
python scripts/benchmark_basic_math_func.py --full-compute --sizes-file scripts/benchmark_sizes_full.json
python scripts/benchmark_optical_algorithm_bench.py --full-compute --sizes-file scripts/benchmark_sizes_full.json
```

## Citations
If you find this code helpeful please cite the code repository via Zenodo: [![DOI](https://zenodo.org/badge/633091061.svg)](https://zenodo.org/doi/10.5281/zenodo.11195111)

## Inspiration
![lina-01](https://user-images.githubusercontent.com/81450894/234684558-71d85349-5bb3-457b-80da-225a37dbd92e.png)



