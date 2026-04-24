"""Core TailCoR computation — Ricci, Tristani, Vergote (2023).

Intuition
---------
TailCoR is a **non-asymptotic, distribution-free** metric that decomposes
pairwise dependence between two random variables (X, Y) into

    - a LINEAR component (what a bivariate Gaussian with the observed
      Pearson correlation would predict), and
    - a NON-LINEAR tail component (the residual — tail contagion that
      Gaussian cannot explain).

The recipe is "a few lines of basic coding" per the paper: rotate (X, Y)
so that the joint is linearly de-correlated and unit-variance, then measure
how much the tail inter-quantile range of the rotated sum exceeds what a
Gaussian would give at the same quantile level.

Mathematically, with empirical Pearson correlation rho between X and Y,
define the projection

    Z = (X + Y) / sqrt(2 * (1 + rho))

Under bivariate Gaussian (X, Y) with correlation rho, Z is standard normal
*for any rho*, so the inter-quantile range IQR_q(Z) equals
Phi^{-1}(q) - Phi^{-1}(1 - q) — the Gaussian reference IQR at that q.

The **composite** TailCoR is the ratio

    S(q) = IQR_q(Z) / ( Phi^{-1}(q) - Phi^{-1}(1 - q) )

- S(q) ≈ 1 under exact bivariate Gaussian (ANY rho)
- S(q) > 1 when the empirical tail is fatter than Gaussian predicts
- S(q) < 1 when the empirical tail is thinner (rare)

The **linear** component is 1.0 by construction (Gaussian reference). The
**non-linear** component is S(q) - 1, so the non-linear value captures the
pure "tail contagion beyond what correlation explains" per Ricci 2023.

Reference
---------
Ricci, L., Tristani, O. & Vergote, O. (2023). "TailCoR". Banco de España
Working Paper 1227. Journal of Empirical Finance.

Notes on convention
-------------------
The paper discusses several rotation choices. v0.1 uses the sum-axis
rotation `(X+Y)/sqrt(2(1+rho))` because it has the cleanest Gaussian
reference ("any rho ⇒ Z is N(0,1)"). Other choices (principal-component,
anti-diagonal `X-Y`, etc.) are equally valid and could be added in v0.2
if caller demand appears.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class TailCoRResult:
    """Decomposition of pairwise dependence at a single quantile level.

    Attributes
    ----------
    q : float
        Tail quantile level in (0.5, 1.0); e.g. 0.95 means IQR_[0.05, 0.95].
    composite : float
        IQR_q(Z) / Gaussian reference IQR. 1.0 under Gaussian, >1 under fat-tail.
    linear : float
        Always 1.0 — the Gaussian reference by construction.
    nonlinear : float
        composite - linear. 0 under Gaussian, >0 under tail contagion.
    rho : float
        Empirical Pearson correlation used in the rotation.
    n : int
        Sample size used.
    """

    q: float
    composite: float
    linear: float
    nonlinear: float
    rho: float
    n: int


def tailcor(
    x: np.ndarray,
    y: np.ndarray,
    q: float = 0.95,
) -> TailCoRResult:
    """Compute the TailCoR linear / non-linear decomposition for a single pair.

    Parameters
    ----------
    x, y : 1-D arrays of equal length
        Paired observations. NaN rows are dropped.
    q : float, in (0.5, 1.0)
        Tail quantile level. 0.95 measures the 5%-tail IQR; 0.99 the 1%-tail.

    Returns
    -------
    TailCoRResult
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"x and y must have same shape, got {x.shape} vs {y.shape}")
    if x.ndim != 1:
        raise ValueError(f"x and y must be 1-D, got shape {x.shape}")
    if not (0.5 < q < 1.0):
        raise ValueError(f"q must be in (0.5, 1.0), got {q}")

    # drop rows where either is NaN
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = x.shape[0]
    if n < 30:
        raise ValueError(f"need >= 30 valid paired observations, got {n}")

    # empirical Pearson correlation; guard the denominator
    rho = float(np.corrcoef(x, y)[0, 1])
    rho = float(np.clip(rho, -0.999, 0.999))

    # rotation: Z = (X + Y) / sqrt(2 * (1 + rho))
    denom = np.sqrt(2.0 * (1.0 + rho))
    # under rho > -1 denom > 0; guarded by clip above
    z = (x + y) / denom

    # empirical tail IQR
    q_hi = float(np.quantile(z, q))
    q_lo = float(np.quantile(z, 1.0 - q))
    iqr_emp = q_hi - q_lo

    # Gaussian reference IQR at same q
    iqr_ref = float(norm.ppf(q) - norm.ppf(1.0 - q))
    if iqr_ref <= 0:
        raise RuntimeError(f"Gaussian reference IQR non-positive at q={q}")

    composite = float(iqr_emp / iqr_ref)
    return TailCoRResult(
        q=float(q),
        composite=composite,
        linear=1.0,
        nonlinear=float(composite - 1.0),
        rho=rho,
        n=n,
    )


def tailcor_profile(
    x: np.ndarray,
    y: np.ndarray,
    quantiles: np.ndarray | None = None,
) -> list[TailCoRResult]:
    """Compute TailCoR across a range of quantile levels.

    Useful for visualising how tail contagion scales from center to extremes.
    Default quantiles go from 0.6 to 0.99 in 8 steps.
    """
    if quantiles is None:
        quantiles = np.array([0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.97, 0.99])
    quantiles = np.asarray(quantiles, dtype=np.float64)
    return [tailcor(x, y, q=float(q)) for q in quantiles]
