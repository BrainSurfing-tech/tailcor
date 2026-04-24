"""Correctness tests for the core TailCoR computation.

Strategy:
  1. Bivariate Gaussian with varying rho → composite ≈ 1, nonlinear ≈ 0.
     This is the FUNDAMENTAL invariant: the Gaussian reference is
     what we're normalising against.
  2. Bivariate Student-t with low df → composite > 1, nonlinear > 0
     (fat-tailed copula → more mass in extremes than Gaussian).
  3. Independence → composite ≈ 1 (rho=0 under Gaussian reference).
  4. Bad-input handling.
"""

import numpy as np
import pytest

from tailcor import tailcor, tailcor_profile
from tailcor.core import TailCoRResult


def _gauss_sample(rng, rho, n):
    """Sample n pairs from bivariate standard normal with correlation rho."""
    cov = np.array([[1.0, rho], [rho, 1.0]])
    return rng.multivariate_normal([0.0, 0.0], cov, size=n).T


def _student_t_copula_sample(rng, rho, df, n):
    """Sample n pairs from a bivariate Student-t with cov rho and df degrees.

    Student-t copula with df=4 has tail dependence — extreme events co-occur
    MORE than a Gaussian copula of the same rho predicts. This is the
    canonical "tail contagion" test case.
    """
    # generate bivariate gaussian with correlation rho
    g = _gauss_sample(rng, rho, n)  # shape (2, n)
    # scale each column by chi-squared multiplier to get Student-t
    chi2 = rng.chisquare(df, size=n)
    scale = np.sqrt(df / chi2)
    return g * scale


def test_gaussian_composite_near_one_across_rho():
    rng = np.random.default_rng(0)
    for rho in [-0.5, -0.2, 0.0, 0.2, 0.5, 0.8]:
        x, y = _gauss_sample(rng, rho, n=10000)
        res = tailcor(x, y, q=0.95)
        assert 0.9 < res.composite < 1.1, (
            f"Gaussian rho={rho} composite {res.composite:.3f} not ~1"
        )
        assert abs(res.nonlinear) < 0.1, (
            f"Gaussian rho={rho} nonlinear {res.nonlinear:.3f} not ~0"
        )


def test_student_t_lowdf_shows_tail_contagion():
    """Student-t with df=4 should produce composite > 1 even at zero rho."""
    rng = np.random.default_rng(1)
    x, y = _student_t_copula_sample(rng, rho=0.0, df=4, n=10000)
    res = tailcor(x, y, q=0.95)
    assert res.composite > 1.15, (
        f"Student-t df=4 composite {res.composite:.3f} not clearly > 1"
    )
    assert res.nonlinear > 0.15


def test_student_t_highdf_looks_more_gaussian():
    """Student-t with df=30 is nearly Gaussian — nonlinear should be small."""
    rng = np.random.default_rng(2)
    x, y = _student_t_copula_sample(rng, rho=0.3, df=30, n=10000)
    res = tailcor(x, y, q=0.95)
    assert abs(res.nonlinear) < 0.15, (
        f"Student-t df=30 nonlinear {res.nonlinear:.3f} too large for near-Gaussian"
    )


def test_linear_is_always_one():
    """Linear component is 1.0 by construction — Gaussian reference."""
    rng = np.random.default_rng(3)
    x, y = _gauss_sample(rng, 0.4, n=5000)
    res = tailcor(x, y, q=0.9)
    assert res.linear == 1.0


def test_rho_recovered():
    rng = np.random.default_rng(4)
    true_rho = 0.6
    x, y = _gauss_sample(rng, true_rho, n=10000)
    res = tailcor(x, y)
    assert abs(res.rho - true_rho) < 0.05


def test_nan_rows_dropped():
    rng = np.random.default_rng(5)
    x, y = _gauss_sample(rng, 0.0, n=1000)
    # inject NaN on 10% of rows
    nan_idx = rng.choice(1000, size=100, replace=False)
    x[nan_idx] = np.nan
    res = tailcor(x, y, q=0.9)
    assert res.n == 900
    assert 0.85 < res.composite < 1.15


def test_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="same shape"):
        tailcor(np.zeros(100), np.zeros(200))


def test_rejects_bad_q():
    rng = np.random.default_rng(7)
    x, y = _gauss_sample(rng, 0.0, n=200)
    with pytest.raises(ValueError, match="\\(0.5, 1.0\\)"):
        tailcor(x, y, q=0.5)
    with pytest.raises(ValueError, match="\\(0.5, 1.0\\)"):
        tailcor(x, y, q=1.0)


def test_rejects_small_samples():
    rng = np.random.default_rng(8)
    x, y = _gauss_sample(rng, 0.0, n=20)
    with pytest.raises(ValueError, match=">= 30"):
        tailcor(x, y)


def test_profile_monotonic_for_tail_copula():
    """For a tail-dependent copula, TailCoR should RISE at more-extreme quantiles.

    Student-t df=4 concentrates mass in the tails, so as q approaches 1 we
    should see the composite grow.
    """
    rng = np.random.default_rng(9)
    x, y = _student_t_copula_sample(rng, rho=0.0, df=4, n=15000)
    profile = tailcor_profile(x, y, quantiles=np.array([0.6, 0.8, 0.95, 0.99]))
    composites = [r.composite for r in profile]
    # composite at q=0.99 should be clearly larger than at q=0.6
    assert composites[-1] > composites[0] + 0.2, (
        f"profile not monotone as expected: {composites}"
    )


def test_result_is_frozen_dataclass():
    rng = np.random.default_rng(10)
    x, y = _gauss_sample(rng, 0.0, n=200)
    res = tailcor(x, y)
    assert isinstance(res, TailCoRResult)
    with pytest.raises(Exception):
        res.composite = 2.0  # type: ignore[misc]
