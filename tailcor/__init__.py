"""tailcor — non-asymptotic, distribution-free decomposition of linear
correlation vs non-linear tail contagion (Ricci, Tristani, Vergote 2023).

Public API:

    from tailcor import tailcor, tailcor_profile, rolling_tailcor

    result = tailcor(x, y, q=0.95)
    result.composite    # ~1 under Gaussian, >1 under tail contagion
    result.nonlinear    # = composite - 1
    result.rho          # empirical Pearson used in rotation

    # quantile sweep:
    profile = tailcor_profile(x, y)

    # rolling over a time series:
    r = rolling_tailcor(x, y, window=252, q=0.95)
    r.composite   # shape (T - window + 1,)
    r.t           # right-edge indices
"""

from tailcor.core import TailCoRResult, tailcor, tailcor_profile
from tailcor.rolling import RollingTailCoR, rolling_tailcor

__all__ = [
    "TailCoRResult",
    "tailcor",
    "tailcor_profile",
    "RollingTailCoR",
    "rolling_tailcor",
]
__version__ = "0.1.0"
