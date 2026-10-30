"""Anytime-valid tests via betting.

To test a candidate mean, play a game against it: start with capital 1,
and on each sample stake a fraction on it landing above (or below) the
candidate. If the candidate is the true mean the capital is a
nonnegative martingale, so P(it ever reaches 2/alpha) <= alpha by
Ville's inequality. Crossing 2/alpha therefore rules the candidate out,
and the guarantee survives peeking after every sample, which is what
the online scheduler needs.

Only the deployed config lives here: capital tracked at the two band
edges +/-eps, aGRAPA bet sizing.
"""

from __future__ import annotations

import math

import numpy as np


class BettingCS:
    """Betting confidence sequence for the mean of samples in [a, b].

    Samples rescale to [0, 1]. Each candidate mean in grid_points runs
    two games: "plus" bets above, "minus" bets below; the candidate is
    ruled out once either log-capital crosses log(2/alpha). Bets are
    sized from past samples only, which is what makes peeking safe.

    aGRAPA stake: gap / (var + gap^2), gap = running mean - candidate.
    Clipped so capital can't hit zero; a bad variance estimate just
    slows the game down.
    """

    def __init__(self, a: float, b: float, alpha: float = 0.05,
                 c: float = 0.9, grid_points: np.ndarray | None = None):
        assert a < b
        assert grid_points is not None and len(grid_points) > 0
        self.a, self.b = a, b
        self.width = b - a
        self.alpha = alpha
        self.log_threshold = math.log(2.0 / alpha)
        self._c = float(c)   # stake cap, as a fraction of the room left

        # candidates in [0,1], nudged off the exact endpoints
        eps = 1e-4
        self.nu_grid = np.clip(np.asarray(sorted(grid_points), dtype=float),
                               eps, 1.0 - eps)
        self.log_K_plus = np.zeros(len(self.nu_grid))
        self.log_K_minus = np.zeros(len(self.nu_grid))

        # running mean/variance (Welford), only used to size bets
        self._n = 0
        self._mean_x = 0.5
        self._m2 = 0.0
        self._var_x = 1.0 / 12.0

    def update(self, d_t: float) -> None:
        if not math.isfinite(d_t):
            return   # keep NaNs out of the running stats
        # Samples outside [a, b] are clipped, so the tested null is "mean of the
        # CLIPPED variable inside the band". An asymmetric tail beyond the [a, b]
        # window shifts that mean relative to the raw mean the pool bootstrap tests.
        # FP-safe (the band-edge A/A nulls carry the real tails); the residual risk
        # is recall-side: such an arm may only claim via the full-budget pool route.
        x_t = min(max((d_t - self.a) / self.width, 0.0), 1.0)

        # capital *= 1 + stake * (sample - candidate), both games at once
        var_safe = max(self._var_x, 1e-6)
        diff = x_t - self.nu_grid
        gap = self._mean_x - self.nu_grid
        base = var_safe + gap * gap
        lam_plus = np.clip(gap / base, 0.0, self._c / self.nu_grid)
        lam_minus = np.clip(-gap / base, 0.0,
                            self._c / (1.0 - self.nu_grid))
        self.log_K_plus += np.log1p(lam_plus * diff)
        self.log_K_minus += np.log1p(-lam_minus * diff)

        self._n += 1
        delta = x_t - self._mean_x
        self._mean_x += (1.0 / self._n) * delta
        self._m2 += delta * (x_t - self._mean_x)
        if self._n >= 2:
            self._var_x = max(self._m2 / self._n, 1e-6)

    def _caps_at(self, point: float) -> tuple[float, float]:
        """(log K+, log K-) at one tracked candidate mean."""
        nu = (point - self.a) / self.width
        hit = np.where(np.abs(self.nu_grid - nu) < 1e-12)[0]
        if not len(hit):
            raise ValueError(f"candidate mean {point} is not tracked by this CS")
        j = int(hit[0])
        return float(self.log_K_plus[j]), float(self.log_K_minus[j])

    def band_log_e(self, eps: float) -> float:
        """log e-value that the change is outside +/-eps; claim at
        >= log(1/alpha). The -log(2) pays for testing both directions."""
        above, _ = self._caps_at(eps)      # betting mean > +eps
        _, below = self._caps_at(-eps)     # betting mean < -eps
        return max(above, below) - math.log(2.0)

    def futile_inside_band(self, eps: float) -> bool:
        """Change is confidently inside +/-eps: too small to matter,
        this arm is never going to claim."""
        _, below_hi = self._caps_at(eps)   # betting mean < +eps
        above_lo, _ = self._caps_at(-eps)  # betting mean > -eps
        return (below_hi >= self.log_threshold
                and above_lo >= self.log_threshold)

    def up_log_e(self, eps: float) -> float:
        """One-sided evidence the mean is above +eps (a slowdown).
        No direction union, so log(2) more power than band_log_e."""
        above, _ = self._caps_at(eps)
        return above

    def futile_below(self, eps: float) -> bool:
        """Mean confidently below +eps, i.e. not a relevant slowdown
        (covers no-change and speedups)."""
        _, below_hi = self._caps_at(eps)
        return below_hi >= self.log_threshold


def claims_cs(alpha: float, eps: float, half: float = 0.06) -> BettingCS:
    """The deployed CS: samples in [-half, +half], capital only at the
    two +/-eps band edges."""
    band_edges = np.array([(-eps + half) / (2 * half),
                           (eps + half) / (2 * half)])
    return BettingCS(-half, half, alpha=alpha, c=0.9, grid_points=band_edges)
