"""Smoke test: the package imports and exposes its version."""

import qdiffusivity


def test_version():
    assert isinstance(qdiffusivity.__version__, str)
    assert len(qdiffusivity.__version__) > 0
