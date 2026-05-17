"""Smoke tests del paquete fraud_detection."""

import pytest


@pytest.mark.unit
def test_package_imports() -> None:
    """El paquete principal debe importarse correctamente."""
    import fraud_detection

    assert fraud_detection.__version__ == "0.1.0"


@pytest.mark.unit
def test_subpackages_importable() -> None:
    """Todos los subpaquetes deben ser importables."""
    from fraud_detection import config, data, evaluation, features, models, pipelines, utils

    assert config is not None
    assert data is not None
    assert features is not None
    assert models is not None
    assert evaluation is not None
    assert pipelines is not None
    assert utils is not None
