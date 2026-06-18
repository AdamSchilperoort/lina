from . import math_module, rt_utils
from . import utils, coro_utils, efc, iefc, aefc, llowfsc, dm

# Top-level backend toggle. Users can do:
#   lina.set_backend("cpu")     # -> numpy + scipy
#   lina.set_backend("gpu")     # -> cupy + cupyx.scipy
#   lina.gpu_available()        # -> True iff cupy is usable
#   lina.get_backend()          # -> "cpu" or "gpu"
from .math_module import (  # noqa: E402, F401
    ensure_np_array,
    get_backend,
    gpu_available,
    set_backend,
)

__version__ = '0.1.0'


