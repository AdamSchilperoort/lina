"""lina_cpp.iefc - re-exports lina.iefc.

See efc.py for the rationale. ``compute_hadamard_scale_factors`` IS in
C++ and is wrapped explicitly so user code that imports it from this
module gets the native version.
"""

from __future__ import annotations

from . import _core
from lina.iefc import (  # noqa: F401
    calibrate,
    init_data,
    make_response_matrix,
    measure_probe_response,
    run,
)


def compute_hadamard_scale_factors(
    had_modes,
    scale_exp=1 / 6,
    scale_thresh=4,
    iwa=2.5,
    owa=13,
    oversamp=4,
    plot=False,
):
    """Native C++ implementation of the Hadamard scale-factor calc."""
    return _core.compute_hadamard_scale_factors(
        had_modes,
        float(scale_exp),
        float(scale_thresh),
        float(iwa),
        float(owa),
        int(oversamp),
        bool(plot),
    )
