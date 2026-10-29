"""Regression tests for the corrected repeated k-fold t-test. Run: python -m pytest test_stats.py (or python test_stats.py)."""
import numpy as np
from model_compare_shap import corrected_repeated_t, holm


def test_all_zero_differences():
    assert corrected_repeated_t(np.zeros(25)) == (0.0, 0.0, 0.0, 1.0)


def test_constant_nonzero_difference_is_reported_not_tested():
    m, lo, hi, p = corrected_repeated_t(np.full(25, 0.01))
    assert (m, lo, hi) == (0.01, 0.01, 0.01) and np.isnan(p)


def test_regular_comparison():
    d = np.random.default_rng(0).normal(0.02, 0.02, 50)
    m, lo, hi, p = corrected_repeated_t(d, k=5, r=10)
    assert lo < m < hi and 0 < p < 1


def test_incomplete_or_nonfinite_inputs_rejected():
    for bad in [np.zeros(24), np.r_[np.zeros(24), np.nan]]:
        try:
            corrected_repeated_t(bad); raise AssertionError("should have failed")
        except ValueError:
            pass


def test_holm_monotone():
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f(); print("ok", name)
