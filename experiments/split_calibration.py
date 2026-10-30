#!/usr/bin/env python3
"""Calibrate the (alpha_claim, alpha_pool) split between the scheduler's two claim
routes (CS stop / pool bootstrap), whose band-edge false-claim rates add. One
scheduler pass per alpha_claim is enough since the trajectory never looks at
alpha_pool; that one gets swept analytically. Split picked on seeds 0-4 under a
margin, then validated on held-out seeds 5-9.

Run: .venv/bin/python experiments/split_calibration.py [tcp8 goduet ...]
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
from adapbench.datasets import EPS_D  # noqa: E402
from adapbench.scheduler import FastScheduler, ReplayRunner  # noqa: E402

ALPHA = 0.01  # the level the UNION has to hold
MARGIN = 0.8  # select at MARGIN*alpha so the held-out check has some slack
BOOT = 800
SIDE = common.SIDE  # ADAPBENCH_SIDE=up -> slowdown-only calibration, _1s outputs
SELECT, HELDOUT = tuple(range(5)), tuple(range(5, 10))
AC_GRID = (0.005, 0.007, 0.010)  # alpha_claim, one scheduler pass each
AP_GRID = (0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 0.010)  # alpha_pool, analytic sweep


_arm_rng = common.arm_rng  # _pool_decide's exact per-arm rng


_load_pools = common.load_pools


def _ref_change(name, pool, seed, band):
    """Fixed-alpha full-run reference verdict, the recall ground truth. Independent
    of the split we're calibrating."""
    r = _arm_rng(name, seed)
    x = pool[r.integers(0, len(pool), size=len(pool))]
    # fresh identically-seeded rng: the same per-arm draw _pool_decide makes
    bm = common.boot_means(x, _arm_rng(name, seed), BOOT)
    return common.boot_change(bm, band, ALPHA, SIDE)


def _boot_claim_vec(r, name, seed, band):
    """Bootstrap 'change' verdict at each alpha_pool, replicating _pool_decide's
    per-arm draw exactly."""
    if len(r) < 2:
        return np.zeros(len(AP_GRID), bool)
    bm = common.boot_means(r, _arm_rng(name, seed), BOOT)
    if SIDE == "up":  # _pool_decide one-sided: lower bound at alpha_pool clears +band
        return np.quantile(bm, list(AP_GRID)) > band
    los = np.quantile(bm, [ap / 2 for ap in AP_GRID])
    his = np.quantile(bm, [1 - ap / 2 for ap in AP_GRID])
    return (los > band) | (his < -band)


def _worker(task):
    """One (mode, edge, seed) pass: run the scheduler per alpha_claim with
    pool_decide off, count CS and bootstrap claims for A/A and for recall."""
    sys.path.insert(0, str(HERE.parent / "src"))
    suite, mode, sign, seed = task
    band = EPS_D
    pools = _load_pools(suite)
    names = list(pools)
    # the DEPLOYED per-arm blocks (incl. BLOCK_PROJ overrides like jctools=6), so the
    # calibration runs the exact shipped config rather than a flat per-suite block
    block = common.block_map(suite, names)
    if mode == "real":
        run_pools = pools
        pos = {a: _ref_change(a, pools[a], seed, band) for a in names}
    else:
        # band-edge null: center each pool, then park the mean on the +/- edge
        run_pools = {a: (pools[a] - pools[a].mean()) + sign * band for a in names}
        pos = {a: False for a in names}
    res = {}
    for ac in AC_GRID:
        budget = {a: len(run_pools[a]) for a in names}
        sched = FastScheduler(run_pools, band=band, alpha_claim=ac, block=block,
                              budget_per_arm=budget, seed=seed, pool_decide=False,
                              side=("up" if SIDE == "up" else None))
        dec, _ = sched.run(ReplayRunner(run_pools, mode="rmit", seed=seed))
        cs = 0; boot = np.zeros(len(AP_GRID))
        cs_p = 0; boot_p = np.zeros(len(AP_GRID)); npos = 0
        for a in names:
            arm = sched.arms[a]
            ispos = pos[a]
            npos += ispos
            if dec[a] == "change":
                cs += 1; cs_p += ispos
            elif dec[a] == "undecided" and arm.spent >= budget[a]:
                # hit budget undecided, so this one falls to the pool route
                v = _boot_claim_vec(np.asarray(arm.raw, float), a, seed, band)
                boot += v
                if ispos:
                    boot_p += v
        res[ac] = dict(n=len(names), cs=cs, boot=boot.tolist(),
                       npos=npos, cs_p=cs_p, boot_p=boot_p.tolist())
    return mode, ("neg" if (sign is not None and sign < 0) else "pos"), seed, res


# --- aggregate and select ---

def _aa(recs, grp):
    """Union A/A per (alpha_claim, alpha_pool) over seed group grp, taken at the
    worse edge: max of the +band and -band rates."""
    out = {}
    for ac in AC_GRID:
        per_edge = {}
        for edge in ("pos", "neg"):
            n = cs = 0; boot = np.zeros(len(AP_GRID))
            for (mode, e, seed), r in recs.items():
                if mode != "null" or e != edge or seed not in grp:
                    continue
                n += r[ac]["n"]; cs += r[ac]["cs"]; boot += np.array(r[ac]["boot"])
            per_edge[edge] = (cs + boot) / max(n, 1)
        out[ac] = np.maximum(per_edge["pos"], per_edge["neg"])
    return out  # ac -> array over AP_GRID


def _recall(recs, grp):
    out = {}
    for ac in AC_GRID:
        npos = cs = 0; boot = np.zeros(len(AP_GRID))
        for (mode, e, seed), r in recs.items():
            if mode != "real" or seed not in grp:
                continue
            npos += r[ac]["npos"]; cs += r[ac]["cs_p"]; boot += np.array(r[ac]["boot_p"])
        out[ac] = (cs + boot) / max(npos, 1)
    return out


def _report(suite, recs, out_dir):
    aa_sel, aa_ho = _aa(recs, SELECT), _aa(recs, HELDOUT)
    rec_sel, rec_ho = _recall(recs, SELECT), _recall(recs, HELDOUT)
    tgt = MARGIN * ALPHA
    # among splits whose select-side worse-edge A/A holds MARGIN*alpha,
    # keep the one with the best select recall
    best = None
    for ac in AC_GRID:
        for j, ap in enumerate(AP_GRID):
            if aa_sel[ac][j] <= tgt:
                cand = (rec_sel[ac][j], ac, ap, j)
                if best is None or cand[0] > best[0]:
                    best = cand
    print(f"\n{suite}{'  [one-sided: slowdowns only]' if SIDE == 'up' else ''}"
          f"  (alpha={ALPHA}, select seeds 0-4 @ A/A<= {tgt:.4f}, held-out 5-9)")
    print(f"  {'a_claim':>8s}{'a_pool':>8s}{'A/A sel':>9s}{'recall sel':>11s}")
    for ac in AC_GRID:
        for j, ap in enumerate(AP_GRID):
            star = " *" if best and (ac, ap) == (best[1], best[2]) else ""
            ok = "" if aa_sel[ac][j] <= tgt else "  (>margin)"
            print(f"  {ac:>8.3f}{ap:>8.3f}{aa_sel[ac][j]:>9.4f}{rec_sel[ac][j]:>11.3f}{ok}{star}")
    if best is None:
        # No pair holds the margin. Fall back to plain alpha: list
        # every pair whose select-side A/A holds ALPHA itself, with its held-out
        # numbers, and mark the deployed split, so the evidence for a no-margin
        # deployment is regenerable from this script instead of a one-off artifact.
        dep = (common.SPLIT_1S if SIDE == "up" else common.SPLIT).get(suite)
        print(f"  no pair holds the {MARGIN:.1f}*alpha margin; pairs at plain alpha, held-out check:")
        print(f"  {'a_claim':>8s}{'a_pool':>8s}{'AA sel':>9s}{'rec sel':>9s}{'AA ho':>9s}{'rec ho':>9s}")
        rows = []
        for ac in AC_GRID:
            for j, ap in enumerate(AP_GRID):
                if aa_sel[ac][j] > ALPHA:
                    continue
                ok = aa_ho[ac][j] <= ALPHA
                mark = "  VALID" if ok else "  INVALID"
                if dep == (ac, ap):
                    mark += "  <- deployed"
                print(f"  {ac:>8.3f}{ap:>8.3f}{aa_sel[ac][j]:>9.4f}{rec_sel[ac][j]:>9.3f}"
                      f"{aa_ho[ac][j]:>9.4f}{rec_ho[ac][j]:>9.3f}{mark}")
                rows.append(dict(alpha_claim=ac, alpha_pool=ap,
                                 aa_sel=float(aa_sel[ac][j]), rec_sel=float(rec_sel[ac][j]),
                                 aa_ho=float(aa_ho[ac][j]), rec_ho=float(rec_ho[ac][j]),
                                 valid_ho=bool(ok)))
        doc = {"suite": suite, "alpha": ALPHA, "side": SIDE, "chosen": None,
               "plain_alpha_pairs": rows}
        if dep is not None:
            hit = next((r for r in rows
                        if (r["alpha_claim"], r["alpha_pool"]) == tuple(dep)), None)
            doc["deployed"] = {"split": list(dep),
                               "aa_ho": None if hit is None else hit["aa_ho"],
                               "valid_ho": bool(hit and hit["valid_ho"])}
        (out_dir / f"split_calibration_{suite}{common.SIDE_TAG}.json").write_text(json.dumps(doc))
        return
    _, ac, ap, j = best
    print(f"  -> CHOSEN split: alpha_claim={ac}, alpha_pool={ap}")
    print(f"     SELECT   : A/A {aa_sel[ac][j]:.4f}   recall {rec_sel[ac][j]:.3f}")
    print(f"     HELD-OUT : A/A {aa_ho[ac][j]:.4f}   recall {rec_ho[ac][j]:.3f}"
          f"   {'VALID (<=alpha)' if aa_ho[ac][j] <= ALPHA else 'INVALID (>alpha)!'}")
    (out_dir / f"split_calibration_{suite}{common.SIDE_TAG}.json").write_text(json.dumps({
        "suite": suite, "alpha": ALPHA, "margin": MARGIN, "side": SIDE,
        "chosen": {"alpha_claim": ac, "alpha_pool": ap},
        "select": {"aa": float(aa_sel[ac][j]), "recall": float(rec_sel[ac][j])},
        "heldout": {"aa": float(aa_ho[ac][j]), "recall": float(rec_ho[ac][j])}}))


def main() -> int:
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    for suite in (sys.argv[1:] or ["tcp8", "goduet"]):
        tasks = ([(suite, "null", +1.0, s) for s in range(10)]
                 + [(suite, "null", -1.0, s) for s in range(10)]
                 + [(suite, "real", None, s) for s in range(10)])
        recs = {}
        with ProcessPoolExecutor(max_workers=8) as ex:
            for mode, edge, seed, res in ex.map(_worker, tasks):
                recs[(mode, edge, seed)] = res
        _report(suite, recs, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
