"""Compatibility wrapper for lina.experimental namespace.

This lets users run code as `import lina_cpp as lina` and still access
`lina.experimental.*` via the lina_cpp package.
"""

from lina.experimental import *  # noqa: F401,F403
