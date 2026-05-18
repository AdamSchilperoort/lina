import numpy as np
import scipy

# cupy is optional. We treat *any* failure during cupy import as "cupy
# is unavailable, fall back to numpy" -- not just plain ImportError.
# In practice cupy raises AttributeError, RuntimeError, OSError or
# similar when its CUDA discovery breaks (e.g. CUDA toolkit missing on
# the host even though the cupy wheel is installed). We don't want
# any of those to kill the whole `import lina` chain.
try:
    import cupy
    import cupyx.scipy
    cupy_avail = True
except Exception as _cupy_err:  # noqa: BLE001
    cupy_avail = False
    cupy = None  # type: ignore[assignment]
    cupyx = None  # type: ignore[assignment]
    import warnings as _warnings
    _warnings.warn(
        f"cupy import failed ({type(_cupy_err).__name__}: {_cupy_err!s}); "
        "falling back to numpy backend. GPU paths will be unavailable.",
        ImportWarning,
        stacklevel=2,
    )

class np_backend:
    """A shim that allows a backend to be swapped at runtime."""
    def __init__(self, src):
        self._srcmodule = src

    def __getattr__(self, key):
        if key == '_srcmodule':
            return self._srcmodule

        return getattr(self._srcmodule, key)
    
class scipy_backend:
    """A shim that allows a backend to be swapped at runtime."""
    def __init__(self, src):
        self._srcmodule = src

    def __getattr__(self, key):
        if key == '_srcmodule':
            return self._srcmodule

        return getattr(self._srcmodule, key)
    
xp = np_backend(cupy) if cupy_avail else np_backend(np)
xcipy = scipy_backend(cupyx.scipy) if cupy_avail else scipy_backend(scipy)

def update_np(module):
    """_summary_

    Parameters
    ----------
    module : _type_
        _description_
    """
    xp._srcmodule = module
    
def update_scipy(module):
    """_summary_

    Parameters
    ----------
    module : _type_
        _description_
    """
    xcipy._srcmodule = module
        
def ensure_np_array(arr):
    if isinstance(arr, np.ndarray):
        return arr
    elif cupy_avail and isinstance(arr, cupy.ndarray):
        return arr.get()
    