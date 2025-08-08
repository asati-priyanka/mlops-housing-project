from src.utils import regression_metrics
import numpy as np


def test_regression_metrics_keys_and_types():
    y_true = np.array([3.0, 2.5, 4.1, 5.0])
    y_pred = np.array([2.8, 2.7, 4.0, 5.2])

    m = regression_metrics(y_true, y_pred)
    # Required keys
    for k in ("rmse", "mae", "r2"):
        assert k in m
        assert isinstance(m[k], float)


def test_regression_metrics_sanity_monotonicity():
    # Perfect prediction -> rmse=0, mae=0, r2=1
    y = np.array([1.0, 2.0, 3.0, 4.0])
    m_perfect = regression_metrics(y, y)
    assert abs(m_perfect["rmse"]) < 1e-12
    assert abs(m_perfect["mae"]) < 1e-12
    assert m_perfect["r2"] == 1.0

    # Worse prediction -> higher rmse/mae, r2 <= 1
    y_pred_bad = np.array([0.0, 0.0, 0.0, 0.0])
    m_bad = regression_metrics(y, y_pred_bad)
    assert m_bad["rmse"] >= m_perfect["rmse"]
    assert m_bad["mae"] >= m_perfect["mae"]
    assert m_bad["r2"] <= 1.0
