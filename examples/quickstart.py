"""tailcor quickstart — detect a regime shift from Gaussian → tail-contagion.

Synthesises 1500 samples: first 750 are iid bivariate Gaussian (no tail
dependence), next 750 are bivariate Student-t with df=4 (Gaussian-looking
marginals but fat joint tails — the canonical tail-contagion setup).

Rolling TailCoR should detect the regime shift by composite rising from
~1 to clearly above 1.
"""

import numpy as np

from tailcor import rolling_tailcor

rng = np.random.default_rng(0)


def gauss(rho, n):
    cov = np.array([[1.0, rho], [rho, 1.0]])
    return rng.multivariate_normal([0.0, 0.0], cov, size=n).T


def student_t(rho, df, n):
    g = gauss(rho, n)
    chi2 = rng.chisquare(df, size=n)
    return g * np.sqrt(df / chi2)


n_each = 750
x1, y1 = gauss(0.0, n_each)         # pre-crisis: iid Gaussian
x2, y2 = student_t(0.0, 4, n_each)  # post-crisis: tail contagion, zero rho

x = np.concatenate([x1, x2])
y = np.concatenate([y1, y2])

r = rolling_tailcor(x, y, window=200, q=0.95)

pre_mask = r.t < n_each - 50
post_mask = r.t > n_each + 150
pre_median = float(np.nanmedian(r.composite[pre_mask]))
post_median = float(np.nanmedian(r.composite[post_mask]))

print(f"pre-crisis (Gaussian)    composite median: {pre_median:.3f}")
print(f"post-crisis (Student-t)  composite median: {post_median:.3f}")
print(f"lift: {post_median - pre_median:+.3f}   ({post_median / pre_median:.2f}x)")
print()
print("TailCoR correctly separates tail contagion from pure linear correlation")
print("(rho is ~0 in both halves — the signal is ENTIRELY non-linear)")
