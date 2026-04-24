"""Rolling-window TailCoR — emit a time series of tail-dependence metrics.

In production settings (crisis early warning, regime-conditioned stress
analysis) you want TailCoR evaluated over a rolling window that keeps
pace with a daily-frequency data feed. This module provides the rolling
wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tailcor.core import TailCoRResult, tailcor


@dataclass(frozen=True)
class RollingTailCoR:
    """Output of a rolling TailCoR computation over a single pair."""

    t: np.ndarray  # right-edge time indices (in sample space)
    composite: np.ndarray  # composite TailCoR per window
    nonlinear: np.ndarray  # non-linear component per window
    rho: np.ndarray  # empirical Pearson per window
    q: float  # tail quantile level used
    window: int  # window size (samples)


def rolling_tailcor(
    x: np.ndarray,
    y: np.ndarray,
    window: int,
    q: float = 0.95,
    min_n: int = 60,
) -> RollingTailCoR:
    """Slide a window of `window` observations across (x, y) and compute TailCoR.

    Parameters
    ----------
    x, y : 1-D arrays, same length T
        Full paired series. NaN handling is delegated to the per-window
        `tailcor` call.
    window : int
        Number of samples per window. Must be >= min_n.
    q : float, in (0.5, 1.0)
        Tail quantile level.
    min_n : int
        Minimum valid-sample count per window. Windows with fewer valid
        (non-NaN) pairs emit NaN instead of raising.

    Returns
    -------
    RollingTailCoR
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError(f"x and y must be equal 1-D arrays, got {x.shape}, {y.shape}")
    if window < min_n:
        raise ValueError(f"window ({window}) must be >= min_n ({min_n})")
    T = x.shape[0]
    if T < window:
        raise ValueError(f"series length {T} < window {window}")

    n_out = T - window + 1
    t_idx = np.arange(window - 1, T)
    composite = np.full(n_out, np.nan)
    nonlinear = np.full(n_out, np.nan)
    rho = np.full(n_out, np.nan)

    for i in range(n_out):
        xw = x[i : i + window]
        yw = y[i : i + window]
        # count valid pairs; if insufficient, keep NaN
        valid = int(np.sum(np.isfinite(xw) & np.isfinite(yw)))
        if valid < min_n:
            continue
        try:
            res = tailcor(xw, yw, q=q)
        except ValueError:
            continue
        composite[i] = res.composite
        nonlinear[i] = res.nonlinear
        rho[i] = res.rho

    return RollingTailCoR(
        t=t_idx,
        composite=composite,
        nonlinear=nonlinear,
        rho=rho,
        q=float(q),
        window=int(window),
    )
