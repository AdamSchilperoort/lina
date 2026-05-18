import numpy as np
import lina_cpp as lina


def test_native_core_imports():
    assert hasattr(lina, "mean")
    assert hasattr(lina, "rms")
    assert hasattr(lina, "pad_or_crop")


def test_mean_rms():
    x = np.arange(9, dtype=np.float64).reshape(3, 3)
    assert np.isclose(lina.mean(x), np.mean(x))
    assert np.isclose(lina.rms(x), np.sqrt(np.mean(x * x)))


def test_gemm_identity():
    a = np.eye(3, dtype=np.float64)
    b = np.arange(6, dtype=np.float64).reshape(3, 2)
    out = lina.gemm(a, b)
    np.testing.assert_allclose(out, b)


def test_pad_or_crop_shape():
    x = np.ones((3, 3), dtype=np.float64)
    y = lina.pad_or_crop(x, 5, 5)
    assert y.shape == (5, 5)
