"""Tests for the AdaBench core: the duet-diff band and the online scheduler
(allocation, anytime-valid claims, certified budget stops)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from adabench.datasets import BIG_REL, EPS_D  # noqa: E402

BAND = EPS_D


def test_eps_d_is_three_percent_in_duet_coords():
    # +-3% relative change maps to EPS_D = 0.03/(2+0.03) in d=(s1-s2)/(s1+s2).
    assert EPS_D == pytest.approx(BIG_REL / (2 + BIG_REL))
    assert EPS_D == pytest.approx(0.0148, abs=1e-4)


def test_fast_scheduler_matches_scheduler():
    # FastScheduler is a drop-in O(log N) replacement: it must produce byte-identical
    # decisions and total spend to the reference O(N) Scheduler.
    from adabench.scheduler import Scheduler, FastScheduler, ReplayRunner
    rng = np.random.default_rng(0)
    pools = {f"reg{i}": rng.normal(3 * BAND, 0.02, 600) for i in range(3)}
    pools.update({f"flat{i}": rng.normal(0.0, 0.03, 600) for i in range(25)})
    d1, s1 = Scheduler(pools, block=4, budget_per_arm=590, seed=0).run(ReplayRunner(pools, seed=0))
    d2, s2 = FastScheduler(pools, block=4, budget_per_arm=590, seed=0).run(ReplayRunner(pools, seed=0))
    assert d1 == d2 and s1 == s2


def test_scheduler_flags_regressions_and_saves_budget():
    from adabench.scheduler import Scheduler, ReplayRunner
    rng = np.random.default_rng(0)
    pools = {f"reg{i}": rng.normal(3 * BAND, 0.02, 800) for i in range(2)}      # regressions
    pools.update({f"flat{i}": rng.normal(0.0, 0.03, 800) for i in range(8)})    # no change
    sched = Scheduler(pools, block=4, budget_per_arm=790, seed=0)
    decisions, spent = sched.run(ReplayRunner(pools, seed=0))
    assert all(decisions[f"reg{i}"] == "change" for i in range(2))   # both regressions flagged
    assert not any(decisions[f"flat{i}"] == "change" for i in range(8))   # no false claim
    assert spent < len(pools) * 790                                  # spent < exhaustive


def test_boot_means_pins_the_calibration_rng_contract():
    # The calibrated alpha split (split_calibration.py) and the full-run reference
    # (common.ref_verdict) are only valid because they replicate _pool_decide's
    # bootstrap draw bit for bit: same 800 resamples, same blake2b per-(seed, arm)
    # rng. Changing Scheduler._boot or the digest string silently invalidates
    # every calibrated split. This test is the tripwire.
    from adabench.scheduler import Scheduler
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
    import common
    rng = np.random.default_rng(7)
    x = rng.normal(0.0, 0.05, 300)
    sched = Scheduler({"trace/arm": x}, budget_per_arm=300, seed=3)
    arm = sched.arms["trace/arm"]
    arm.raw.extend(x.tolist())          # what a full replay would have deposited
    want = common.boot_means(x, common.arm_rng("trace/arm", 3), common.BOOT)
    assert sched._boot == common.BOOT == 800
    assert np.array_equal(sched._boot_means(arm), want)


def test_certified_stop_retires_a_hopeless_arm_within_budget():
    # An arm sitting exactly on zero effect with high variance can neither claim
    # nor pass futility fast; the certified budget stop must retire it as
    # undecided before the budget is exhausted, and must never claim.
    from adabench.scheduler import Scheduler, ReplayRunner
    rng = np.random.default_rng(1)
    v = rng.normal(0.0, 0.25, 4000)
    v = v - v.mean()                                   # dead-centered null
    pools = {"hopeless": v}
    sched = Scheduler(pools, block=4, budget_per_arm=4000, seed=0)
    decisions, spent = sched.run(ReplayRunner(pools, seed=0))
    assert decisions["hopeless"] != "change"
    assert spent < 4000
