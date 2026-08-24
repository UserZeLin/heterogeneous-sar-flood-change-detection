import numpy as np

from hsarfcd.change_detection import classify_gmm, clean_mask, difference_magnitude
from hsarfcd.metrics import binary_metrics


def test_gmm_selects_high_difference_component():
    before = np.zeros((3, 64, 64), dtype=np.float32)
    after = before.copy()
    after[:, 20:44, 20:44] = 1.0
    magnitude = difference_magnitude(before, after)
    raw, probability, model = classify_gmm(magnitude, max_samples=10_000, random_state=3)
    assert raw[30, 30]
    assert not raw[0, 0]
    assert probability[30, 30] > probability[0, 0]
    assert model.means_.shape == (2, 1)


def test_morphology_removes_isolated_pixel():
    mask = np.zeros((32, 32), dtype=bool)
    mask[8:20, 8:20] = True
    mask[1, 1] = True
    cleaned = clean_mask(mask, opening_radius=0, closing_radius=0, min_object_size=10)
    assert cleaned[10, 10] == 1
    assert cleaned[1, 1] == 0


def test_binary_metrics_perfect_prediction():
    reference = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    metrics = binary_metrics(reference, reference.copy())
    assert metrics["f1"] == 1.0
    assert metrics["iou"] == 1.0
    assert metrics["overall_accuracy"] == 1.0

