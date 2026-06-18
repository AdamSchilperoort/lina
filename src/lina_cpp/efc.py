"""lina_cpp.efc - re-exports lina.efc.

Electric field conjugation. The math hotspots (gemv, gemm, svd,
mft_forward, etc.) inside the EFC loop already go through C++ when the
caller's wavefront pipeline uses ``lina_cpp.props`` / ``lina_cpp.utils``.
``efc.run`` itself is Python control flow and re-exports here.

A native C++ ``lina::efc::run`` lives in ``cpp/src/efc.cpp`` for the
``lina_runner`` benchmarking tool; it is *not* currently exposed via
pybind because the public ``lina.efc.run(efc_data, ...)`` signature
takes a Python ``efc_data`` dict that's tied to the Python-side pwp /
shmim glue. Exposing an equivalent pybind wrapper is a follow-up.
"""

from __future__ import annotations

from lina_cpp._pyref.efc import (  # noqa: F401
    calibrate,
    init_data,
    run,
)
