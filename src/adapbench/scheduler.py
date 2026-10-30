"""AdapBench online scheduler (paper Algorithm 1): the deployable prototype.

Given a suite of benchmarks ("arms") that just ran against a code change,
find the ones that got slower or faster with as few measurements as
possible. One loop, one measurement at a time:

  1. allocate - pick the active arm closest to a decision (plus a bit of
     round-robin exploration)
  2. measure  - one sample, via a pluggable Runner
  3. update   - fold it into the arm's betting CS (cs.py), block-averaged
     so heavy tails can't break the statistics
  4. decide   - retire on the earliest of: claim (change outside the
     +/-band), futility (inside, too small to matter), or a certified
     budget stop (even an optimistic reading can't reach a claim within
     budget, confirmed at two consecutive checkpoints)

No historical data needed. State is O(1) per arm; selection is O(#active)
here, O(log #arms) in FastScheduler. Measurement is pluggable: implement
Runner.measure and the scheduler works unchanged.

    from adapbench.scheduler import Scheduler, ReplayRunner
    sched = Scheduler(pools.keys(), block=4, budget_per_arm=1490)
    decisions, spent = sched.run(ReplayRunner(pools))
"""
from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass, field
from typing import Protocol

from .cs import claims_cs
from .datasets import EPS_D

# Certified-stop checkpoints: (fraction of budget, z). A stop needs two
# consecutive failed checkpoints. Late checkpoints relax z: the estimate
# is tight by then, so the softer z adds stops at ~zero recall cost.
SCHED_CS = ((0.5, 1.64), (0.75, 1.64), (0.9, 1.28), (0.97, 1.0))


class Runner(Protocol):
    """One duet-diff sample per call; None when the arm is exhausted."""

    def measure(self, arm: str) -> float | None:
        ...


@dataclass
class _Arm:
    """Per-benchmark state. nb/sm/sq give mean and variance in O(1)."""
    name: str
    block: int                                    # samples averaged per block
    cs: object                                    # betting confidence sequence
    budget: int | None = None                     # measurement cap / certified-stop horizon
    buf: list = field(default_factory=list)       # samples in the current block
    raw: list = field(default_factory=list)       # all finite samples (for the bootstrap)
    nb: int = 0                                   # completed blocks
    sm: float = 0.0                               # sum of block means
    sq: float = 0.0                               # sum of squared block means
    spent: int = 0                                # raw measurements taken
    active: bool = True
    decision: str = "undecided"                   # 'change' | 'no_change' | 'undecided'
    _pc: int = 0                                  # next pool-checkpoint index (opt-in)
    _cp: int = 0                                  # next certified-stop checkpoint index
    _cp_bad: bool = False                         # previous checkpoint said "hopeless"
    _cp_jj: int = -1                              # depth of the previous checkpoint look

    def mean(self) -> float:
        return self.sm / self.nb

    def var(self) -> float:
        # floored: a zero variance would blow up the divisions below
        m = self.sm / self.nb
        return max(self.sq / self.nb - m * m, 1e-9)


class Scheduler:
    """The allocate-measure-update-decide loop (see module docstring).

    Main knobs: band = "big enough to matter"; block = samples averaged
    before the stats see them (int or per-arm dict, from the A/A gate);
    budget_per_arm = per-arm cap and certified-stop horizon (int, dict,
    or None); side='up' = slowdown-only (more power, speedups retire
    cheaply).

    Two alphas because claims fire two ways (the anytime CS mid-run, or
    the fixed-n bootstrap at budget) on disjoint sample paths, so their
    false-claim rates add. alpha_claim budgets the CS, alpha_pool the
    bootstrap; both default to alpha, and the calibrated split
    (experiments/split_calibration.py) keeps the union <= alpha.
    """

    def __init__(self, arms, *, band: float = EPS_D, alpha: float = 0.01,
                 alpha_claim: float | None = None, alpha_pool: float | None = None, block=1,
                 budget_per_arm=None, ef: float = 0.15, minn: int = 3,
                 sched=SCHED_CS, seed: int = 0,
                 pool_decide: bool = True, side: str | None = None,
                 pool_checkpoints=None):
        # Opt-in extension, off by default (default path byte-identical
        # to the deployed scheduler; verified by tests):
        #   pool_checkpoints - ((frac, level), ...): at frac*budget spent, a
        #                      bootstrap CI at `level` may claim or retire
        #                      no_change. Kills the "undecided arms run to
        #                      full budget" cliff. level <= 0 = confirm-only
        #                      (never claims, so no alpha); claim levels must
        #                      come from the alpha-split calibration.
        ac = alpha if alpha_claim is None else alpha_claim    # CS budget
        ap = alpha if alpha_pool is None else alpha_pool      # bootstrap budget
        self.band = band
        self.thr = math.log(1.0 / ac)             # claim threshold, log e-value
        self.side = side
        self.ef = ef
        self.minn = minn
        self.sched = sched
        self._rng = random.Random(seed)
        self._pool = pool_decide
        self._alpha = ap
        self._boot = 800                          # bootstrap resamples
        self._pool_seed = seed
        self._pcs = tuple(pool_checkpoints) if pool_checkpoints else None
        self.arms = {
            a: _Arm(a,
                    block[a] if isinstance(block, dict) else block,
                    claims_cs(ac, eps=band),
                    budget=budget_per_arm[a] if isinstance(budget_per_arm, dict)
                           else budget_per_arm)
            for a in arms
        }

    # --- allocation ---------------------------------------------------

    def _score(self, arm: _Arm) -> float:
        """Priority, lowest wins.

        Cold (< minn blocks): no distance estimate yet, prefer the biggest
        apparent effect. Warm: estimated blocks until this arm can claim.
        Capital grows ~ gap^2 / (2*var) per block once the mean sits `gap`
        outside the band, and a claim needs log(1/alpha) of it. A mean
        inside the band can't claim; park it at the back."""
        if arm.nb < self.minn:
            return -(abs(arm.mean()) if arm.nb else 0.0)
        gap = abs(arm.mean()) - self.band
        return (2 * arm.var() * self.thr / (gap * gap)) if gap > 0 else 1e18

    def _select(self) -> _Arm | None:
        """Best-scoring arm; a fraction ef of the time the least-measured
        one, so nothing is starved."""
        active = [a for a in self.arms.values() if a.active]
        if not active:
            return None
        if self._rng.random() < self.ef:
            return min(active, key=lambda a: a.nb)      # explore
        return min(active, key=self._score)             # exploit

    # --- update + decide ----------------------------------------------

    @staticmethod
    def _finish(arm: _Arm, verdict: str) -> None:
        arm.active = False
        arm.decision = verdict

    def _observe(self, arm: _Arm, x: float) -> None:
        """Fold one sample into the arm; decide if anything fires."""
        arm.spent += 1
        if not math.isfinite(x):
            return   # NaN would poison the running sums; CS rejects it too
        arm.raw.append(x)   # kept for the bootstrap at budget

        if self._pcs is not None and arm.budget is not None:
            self._ckpt_step(arm)                        # opt-in checkpoints
            if not arm.active:
                return

        arm.buf.append(x)
        if len(arm.buf) == arm.block:                   # block complete
            self._block_step(arm)

    def _ckpt_step(self, arm: _Arm) -> None:
        """Run every pool checkpoint the arm's spend has reached."""
        while (arm._pc < len(self._pcs)
               and arm.spent >= math.ceil(self._pcs[arm._pc][0] * arm.budget)):
            self._ckpt_decide(arm, self._pcs[arm._pc][1])
            arm._pc += 1
            if not arm.active:
                return

    def _block_step(self, arm: _Arm) -> None:
        """Fold the completed block and ask the CS for a verdict."""
        block_mean = sum(arm.buf) / len(arm.buf)
        arm.buf = []
        arm.nb += 1
        arm.sm += block_mean
        arm.sq += block_mean * block_mean
        arm.cs.update(float(block_mean))

        # futility needs some history first, hence nb >= 8
        if self.side == "up":       # slowdowns only
            claim = arm.cs.up_log_e(self.band) >= self.thr
            futile = arm.nb >= 8 and arm.cs.futile_below(self.band)
        else:                       # two-sided (default)
            claim = arm.cs.band_log_e(self.band) >= self.thr
            futile = arm.nb >= 8 and arm.cs.futile_inside_band(self.band)

        if claim:
            self._finish(arm, "change")
        elif futile:
            self._finish(arm, "no_change")
        elif arm.budget is not None:
            self._certified(arm)

    def _certified(self, arm: _Arm) -> None:
        """Certified budget stop. At each SCHED_CS checkpoint, take an
        optimistic effect (|mean| + z standard errors), project capital
        growth at that effect over the budget, and check whether a claim
        is still reachable; hopeless at two consecutive checkpoints ->
        stop as undecided. Only stops, never claims: costs recall, never
        alpha. Judges fresh projections, not realized capital: an
        unlucky early run must not kill an arm on its way to a claim."""
        horizon = max(1, arm.budget // arm.block)       # budget in blocks
        while arm._cp < len(self.sched):
            frac, z = self.sched[arm._cp]
            depth = max(8, int(frac * horizon))
            if arm.nb < depth:
                return   # checkpoint not reached yet
            arm._cp += 1
            if depth >= horizon:
                break
            if depth == arm._cp_jj:
                continue   # tiny budget: one look must not confirm itself
            m = arm.mean() if self.side == "up" else abs(arm.mean())
            var = arm.var()
            optimistic_gap = m + z * math.sqrt(var / arm.nb) - self.band
            growth = 0.0 if optimistic_gap <= 0 else \
                optimistic_gap * optimistic_gap / (2 * var)
            # bar: log(1/alpha) one-sided, log(2/alpha) two-sided (union)
            bar = self.thr if self.side == "up" else self.thr + math.log(2.0)
            hopeless = horizon * growth < bar
            if hopeless and arm._cp_bad:
                self._finish(arm, "undecided")
                return
            arm._cp_bad = hopeless
            arm._cp_jj = depth

    # --- the bootstrap (mid-run checkpoints + the fallback at budget) ---

    def _boot_means(self, arm: _Arm, tag: str = ""):
        """Bootstrap means of the arm's raw samples, from a deterministic
        per-(seed, arm, tag) RNG, independent of scheduling order and
        reproducible by an external reference reading the same stream."""
        import hashlib
        import numpy as np
        r = np.asarray(arm.raw, float)
        h = hashlib.blake2b(f"{self._pool_seed}|{arm.name}{tag}".encode(),
                            digest_size=8).digest()
        rng = np.random.default_rng(int.from_bytes(h, "big"))
        return r[rng.integers(0, len(r), size=(self._boot, len(r)))].mean(1)

    def _ckpt_decide(self, arm: _Arm, level: float) -> None:
        """One pool checkpoint: bootstrap CI on the raw samples so far.
        Clears the band -> claim; strictly inside -> retire no_change
        (costs recall risk, never alpha); else keep running. level <= 0 =
        confirm-only (never claims), CI at the nominal pool alpha."""
        import numpy as np
        if len(arm.raw) < 8:
            return
        boot_means = self._boot_means(arm, f"|ckpt{arm._pc}")
        ci_level = self._alpha if level <= 0 else level
        if self.side == "up":
            lo, hi = np.percentile(boot_means, [100 * ci_level, 100 * (1 - ci_level)])
            if level > 0 and lo > self.band:
                self._finish(arm, "change")
            elif hi < self.band:                        # confidently not a slowdown
                self._finish(arm, "no_change")
            return
        lo, hi = np.percentile(boot_means, [100 * ci_level / 2, 100 * (1 - ci_level / 2)])
        if level > 0 and (lo > self.band or hi < -self.band):
            self._finish(arm, "change")
        elif lo > -self.band and hi < self.band:        # confidently inside the band
            self._finish(arm, "no_change")

    def _pool_decide(self, arm: _Arm) -> None:
        """Arm hit budget without a CS verdict: fixed-n bootstrap CI over
        all its raw samples at alpha_pool. Fixed-n beats the anytime CS
        at equal n, so this rescues borderline regressions; the slight
        band-edge over-claim is what the alpha split keeps bounded."""
        import numpy as np
        if len(arm.raw) < 2:
            arm.decision = "no_change"
            return
        boot_means = self._boot_means(arm)
        if self.side == "up":                           # slowdowns only
            lo = np.percentile(boot_means, 100 * self._alpha)
            arm.decision = "change" if lo > self.band else "no_change"
        else:
            lo, hi = np.percentile(boot_means, [100 * self._alpha / 2,
                                                100 * (1 - self._alpha / 2)])
            arm.decision = "change" if (lo > self.band or hi < -self.band) else "no_change"

    def _retire(self, arm: _Arm) -> None:
        """Deactivate at end of run; undecided arms go to the bootstrap."""
        arm.active = False
        if self._pool and arm.decision == "undecided":
            self._pool_decide(arm)

    # --- main loop ----------------------------------------------------

    # FastScheduler hooks; the base scheduler needs no bookkeeping.
    def _arm_done(self, arm: _Arm) -> None:
        pass

    def _arm_measured(self, arm: _Arm, nb_before: int, was_active: bool) -> None:
        pass

    def run(self, runner: Runner):
        """Run until no arm is active.
        Returns ({arm: decision}, total_spent); fills detected_at =
        {arm: suite measurements spent when first flagged 'change'}."""
        self.detected_at: dict = {}
        spent = 0
        while True:
            arm = self._select()
            if arm is None:
                break
            if arm.budget is not None and arm.spent >= arm.budget:
                self._retire(arm)                       # hit its own budget
                self._arm_done(arm)
            else:
                x = runner.measure(arm.name)
                if x is None:                           # runner exhausted it
                    self._retire(arm)
                    self._arm_done(arm)
                else:
                    spent += 1
                    nb_before, was_active = arm.nb, arm.active
                    self._observe(arm, x)
                    self._arm_measured(arm, nb_before, was_active)
            if arm.decision == "change" and arm.name not in self.detected_at:
                self.detected_at[arm.name] = spent
        return {a.name: a.decision for a in self.arms.values()}, spent


class FastScheduler(Scheduler):
    """Same policy and decisions, O(log #arms) per step: the version to
    run at suite scale.

    Only _select changes: two lazy heaps (by score / by block count)
    replace the linear argmin. Key changes push a fresh entry instead of
    updating in place; a generation counter marks old entries stale, and
    stale/inactive tops are dropped on pop. Only the just-measured arm
    ever re-keys. Decisions are identical to Scheduler (tested)."""

    def _init_heaps(self) -> None:
        # entry = (key, arm_index, generation, name); live iff the arm is
        # active and the generation matches. arm_index tiebreak reproduces
        # min()'s insertion-order behavior exactly.
        self._idx = {a.name: i for i, a in enumerate(self.arms.values())}
        self._gen = {a.name: 0 for a in self.arms.values()}
        self._sheap: list = []                          # by score (exploit)
        self._nheap: list = []                          # by nb (explore)
        for a in self.arms.values():
            self._push(a)
        self._nact = len(self.arms)                     # active-arm count

    def _push(self, arm: _Arm) -> None:
        self._gen[arm.name] += 1
        g, i = self._gen[arm.name], self._idx[arm.name]
        heapq.heappush(self._sheap, (self._score(arm), i, g, arm.name))
        heapq.heappush(self._nheap, (arm.nb, i, g, arm.name))

    def _select(self) -> _Arm | None:
        if self._nact == 0:
            return None
        heap = self._nheap if self._rng.random() < self.ef else self._sheap
        while heap:
            _, _, g, name = heap[0]
            arm = self.arms[name]
            if arm.active and g == self._gen[name]:
                return arm
            heapq.heappop(heap)                         # stale or retired
        return None

    def _arm_done(self, arm: _Arm) -> None:
        self._nact -= 1

    def _arm_measured(self, arm: _Arm, nb_before: int, was_active: bool) -> None:
        if was_active and not arm.active:               # decided this step
            self._nact -= 1
        elif arm.nb != nb_before:                       # block done: re-key
            self._push(arm)

    def run(self, runner: Runner):
        self._init_heaps()
        return super().run(runner)


# --- replaying recorded pools ---------------------------------------------

def hier_perm(rng, n, sizes):
    """Hierarchical cluster permutation of [0..n): cut into contiguous
    blocks of sizes[0] (ragged tail kept), shuffle the block order,
    recurse with sizes[1:]; below the last level shuffle elements.
    Randomizes exactly what a fresh RMIT run would (instance/suite-run/
    iteration order) while keeping within-cluster dependence intact."""
    import numpy as np

    def rec(seg, lvl):
        if lvl == len(sizes):
            return rng.permutation(seg)
        b = sizes[lvl]
        blocks = [seg[k:k + b] for k in range(0, len(seg), b)]
        order = rng.permutation(len(blocks))
        return np.concatenate([rec(blocks[o], lvl + 1) for o in order])

    return rec(np.arange(n), 0)


# Cluster sizes for 'hier' replay of an RMIT (3,3,5,5)-style collection:
# ~75 samples per instance run, 25 per suite run, 5 per iteration.
HIER_SIZES = (75, 25, 5)


def window_perm(rng, n, w=25):
    """Shuffle within consecutive w-sample windows, keep window order.
    This is the live-interleaving model: local order random, time still
    advances. Sits between 'ordered' and 'shuffle'."""
    import numpy as np
    return np.concatenate([lo + rng.permutation(min(w, n - lo)) for lo in range(0, n, w)])


class ReplayRunner:
    """Replays each arm's recorded pool. Modes:

    * 'rmit'    - i.i.d. with replacement (idealized independent collection)
    * 'shuffle' - each real sample once, random order: the honest replay of
      an RMIT pool (no bootstrap duplication, no recorded-order confound)
    * 'hier'    - each sample once, hierarchical cluster permutation
      (hier_perm): faithful simulation of re-running RMIT
    * 'window'  - each sample once, shuffled within local windows
      (window_perm): the live-interleaving model
    * 'ordered' - recorded order; None past the end
    """

    def __init__(self, pools: dict, mode: str = "rmit", seed: int = 0):
        import hashlib
        import numpy as np
        self.mode = mode
        self._pools = {a: np.asarray(p, float) for a, p in pools.items()}
        self._i = {a: 0 for a in pools}

        # per-arm RNG from a stable hash of (seed, arm); Python's hash()
        # is salted per process and would break reproducibility
        def _seed(a: str) -> int:
            h = hashlib.blake2b(f"{seed}|{a}".encode(), digest_size=8).digest()
            return int.from_bytes(h, "big")
        self._rng = {a: np.random.default_rng(_seed(a)) for a in pools}

        # permutations drawn up front as each rng's first consumption, so a
        # fresh identically-seeded rng reproduces them (the external
        # reference stream relies on this)
        if mode == "shuffle":
            self._perm = {a: self._rng[a].permutation(len(self._pools[a])) for a in pools}
        elif mode == "window":
            self._perm = {a: window_perm(self._rng[a], len(self._pools[a])) for a in pools}
        elif mode == "hier":
            self._perm = {a: hier_perm(self._rng[a], len(self._pools[a]), HIER_SIZES)
                          for a in pools}

    def measure(self, arm: str) -> float | None:
        p = self._pools[arm]
        if self.mode in ("ordered", "shuffle", "window", "hier"):
            i = self._i[arm]
            if i >= len(p):
                return None
            self._i[arm] += 1
            return float(p[i] if self.mode == "ordered" else p[self._perm[arm][i]])
        return float(p[int(self._rng[arm].integers(0, len(p)))])   # rmit


if __name__ == "__main__":                        # tiny demo on synthetic pools
    import numpy as np
    rng = np.random.default_rng(0)
    band = EPS_D
    pools = {f"reg{i}": rng.normal(3 * band, 0.02, 1500) for i in range(3)}       # regressions
    pools |= {f"flat{i}": rng.normal(0.0, 0.03, 1500) for i in range(20)}         # no change
    sched = Scheduler(pools, block=4, budget_per_arm=1490, seed=0)
    decisions, spent = sched.run(ReplayRunner(pools, seed=0))
    changed = [a for a, d in decisions.items() if d == "change"]
    exhaustive = len(pools) * 1490
    print(f"decided {len(decisions)} benchmarks in {spent} measurements "
          f"({100 * (1 - spent / exhaustive):.0f}% below the {exhaustive} exhaustive); "
          f"flagged as changed: {sorted(changed)}")
