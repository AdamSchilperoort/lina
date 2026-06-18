"""Shared pytest configuration for the lina_cpp parity suite."""


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: end-to-end control-model / iEFC parity checks (build a small "
        "optical model; deselect with -m 'not slow')",
    )
