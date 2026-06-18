"""lina_cpp.rt_utils - real-time / shmim helpers (re-exported from lina).

Most of these helpers wrap MagAOX shmim streams (ImageStreamIO) and
INDI telemetry RPC. The C++ library DOES contain a fast shmim writer
(``cpp/src/shmim_utils.cpp``), but the Python ``lina.rt_utils.write`` is
already a one-liner around ``magpyx`` so there is no measurable perf
benefit to wrapping it. We re-export until the closed-loop runtime
(``llowfsc.run``) is ported to C++ -- at which point the C++ side will
write to shmim directly via ``lina::shmim_write_*`` and bypass these
helpers entirely.
"""

from __future__ import annotations

from lina_cpp._pyref.rt_utils import (  # noqa: F401
    Process,
    change_shmim_permissions,
    create_shmim,
    read_telem_data,
    read_telem_times,
    stack,
    toggle_telem,
    unpack_data,
    write,
    write_dm,
    zero,
)
