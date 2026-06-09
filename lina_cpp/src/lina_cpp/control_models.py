"""Compatibility wrapper for lina.control_models.

The 1-DM high-level control model is still Python-level logic, so expose it
through lina_cpp for API parity when users swap
    import lina      as lina
    import lina_cpp  as lina
"""

from lina.control_models import *  # noqa: F401,F403
