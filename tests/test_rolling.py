"""Tests for rolling-window TailCoR."""

import numpy as np
import pytest

from tailcor import rolling_tailcor


def _gauss_sample(rng, rho, n):
    cov = np.array([[1.0, rho], [rho, 1.0]])
    return rng.multivariate_normal([0.0, 0.0], cov, size=n).T


def _student_t_copula_sample(rng, rho, df, n):
    g = _gauss_sample(rng, rho, n)
    chi2 = rng.chisquare(df, size=n)
    scale = np.sqrt(df / chi2)
    return g * scale


def test_rolling_shape_and_nan_structure():
    rng = np.random.default_rng(0)
    x, y = _gauss_sample(rng, 0.0, n=500)
    r = rolling_tailcor(x, y, window=100, q=0.9)
    assert r.composite.shape == (401,)
    assert r.t.shape == (401,)
    # first window ends at index 99 (window-1)
    assert r.t[0] == 99
    assert r.t[-1] == 499
    # Gaussian composite should hover around 1 across all windows
    clean = r.composite[np.isfinite(r.composite)]
    assert clean.size > 0
    assert 0.8 < np.median(clean) < 1.2


def test_rolling_detects_regime_shift():
    """First half: iid Gaussian (no tail contagion).
    Second half: Student-t(df=4) with tail contagion.
    Rolling composite should clearly rise in the second half."""
    rng = np.random.default_rng(1)
    n = 1500
    x1, y1 = _gauss_sample(rng, 0.0, n=n // 2)
    x2, y2 = _student_t_copula_sample(rng, rho=0.0, df=4, n=n // 2)
    x = np.concatenate([x1, x2])
    y = np.concatenate([y1, y2])

    r = rolling_tailcor(x, y, window=200, q=0.95)
    # pre-regime-shift windows: composite ~1
    # post-regime-shift windows: composite > 1.1
    pre = r.composite[r.t < n // 2 - 50]
    post = r.composite[r.t > n // 2 + 150]
    pre_clean = pre[np.isfinite(pre)]
    post_clean = post[np.isfinite(post)]
    assert pre_clean.size > 10 and post_clean.size > 10
    assert np.median(post_clean) > np.median(pre_clean) + 0.1, (
        f"no regime-shift detected: pre median={np.median(pre_clean):.3f} "
        f"post median={np.median(post_clean):.3f}"
    )


def test_rolling_rejects_window_smaller_than_min_n():
    rng = np.random.default_rng(2)
    x, y = _gauss_sample(rng, 0.0, n=200)
    with pytest.raises(ValueError, match="min_n"):
        rolling_tailcor(x, y, window=40, min_n=60)


def test_rolling_rejects_series_shorter_than_window():
    rng = np.random.default_rng(3)
    x, y = _gauss_sample(rng, 0.0, n=80)
    with pytest.raises(ValueError, match="length"):
        rolling_tailcor(x, y, window=100)
