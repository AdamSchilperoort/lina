"""lina_cpp.dm - C++-backed mirror of lina.dm.

DM helper functions for building command/probe modes. Anything with a
native C++ implementation is wrapped here; the rest re-exports from
lina.dm.
"""

from __future__ import annotations

import numpy as np

from . import _core
from .math_module import ensure_np_array, xp


def create_mask(Nact=34, return_np=False):
    """Boolean DM actuator mask."""
    out = _core.create_mask(int(Nact))
    return out if return_np else xp.asarray(out)


def make_gaussian_inf_fun(*args, **kwargs):
    """Influence-function kernel for a Gaussian-shaped DM actuator."""
    return xp.asarray(_core.make_gaussian_inf_fun(*args, **kwargs))


def create_hadamard_modes(dm_mask, return_np=False):
    """Hadamard mode basis on the DM."""
    dm_mask_np = ensure_np_array(dm_mask)
    had = _core.create_hadamard_modes(
        dm_mask_np.astype(np.uint8, copy=False)
    )

    # Parity with lina.dm.create_hadamard_modes: return (Nmodes, Nact, Nact),
    # not flattened (Nmodes, Nact*Nact).
    nact = int(dm_mask_np.shape[0])
    if had.ndim == 2 and had.shape[1] == nact * nact:
        had = had.reshape(had.shape[0], nact, nact)
    return had if return_np else xp.asarray(had)


def create_fourier_modes(*args, **kwargs):
    """2D-Fourier mode basis on the DM (sin & cos)."""
    return xp.asarray(_core.create_fourier_modes(*args, **kwargs))


def make_fourier_command(x_cpa=10, y_cpa=10, Nact=34, phase=0, return_np=False):
    """Single Fourier-mode DM command."""
    out = _core.make_fourier_command(int(x_cpa), int(y_cpa), int(Nact), float(phase))
    return out if return_np else xp.asarray(out)


def make_f(h=10, w=6, shift=(-1, 0), Nact=34, return_np=False):
    """The 'F' shape (used as a calibration probe). Pure C++."""
    out = _core.make_f(int(h), int(w), int(shift[0]), int(shift[1]), int(Nact))
    return out if return_np else xp.asarray(out)


def make_ring(rad=15, Nact=34, thresh=1 / 2):
    """A unit-amplitude ring at radius `rad` in actuator space."""
    return xp.asarray(_core.make_ring(float(rad), int(Nact), float(thresh)))


def make_cross_command(xc=[0], yc=[0], Nact=34):
    """A cross-shaped DM command."""
    return xp.asarray(_core.make_cross_command(list(xc), list(yc), int(Nact)))


# Functions not yet ported -- defer to lina.dm.
from lina_cpp._pyref.dm import (  # noqa: E402, F401
    create_all_poke_modes,
    create_fourier_probes,
)
