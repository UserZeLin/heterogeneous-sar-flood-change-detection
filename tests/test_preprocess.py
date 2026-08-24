import numpy as np

from hsarfcd.inference import tile_origins
from hsarfcd.preprocess import power_to_db
from hsarfcd.raster import NormalizationSpec, normalize, repeat_to_channels


def test_power_to_db():
    values = np.array([[[1.0, 10.0, 100.0]]], dtype=np.float32)
    np.testing.assert_allclose(power_to_db(values), [[[0.0, 10.0, 20.0]]], atol=1e-5)


def test_percentile_normalization_range():
    values = np.arange(100, dtype=np.float32).reshape(1, 10, 10)
    result, stats = normalize(values, NormalizationSpec(lower=0, upper=100))
    assert result.min() == -1.0
    assert result.max() == 1.0
    assert len(stats) == 2


def test_repeat_single_band():
    values = np.ones((1, 4, 4), dtype=np.float32)
    assert repeat_to_channels(values).shape == (3, 4, 4)


def test_tile_origins_cover_last_pixel():
    origins = tile_origins(1000, tile_size=256, overlap=32)
    assert origins[0] == 0
    assert origins[-1] == 744
    assert all(left < right for left, right in zip(origins, origins[1:]))
