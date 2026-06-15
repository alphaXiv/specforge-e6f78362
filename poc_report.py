#!/usr/bin/env python3
"""Parse CE vs TV training logs across multiple seeds and emit the Bebop PoC
comparison (EVAL.md).

Reads the per-MTP-step rejection-sampling acceptance rates that
``train_eagle3.py`` prints at eval time (lines tagged ``[mode=eval]``), keeps
the final eval block from each per-seed run, and writes a CE-vs-TV table with
per-position mean ± std across seeds plus a paired one-sided t-test on the
per-seed mean acceptance. The core claim of arXiv:2606.12370 is that the
TV-trained head reaches >= CE acceptance at matched training; the verdict is
"REPRODUCED" only when the paired one-sided test rejects H0 at p < 0.05.
"""
import argparse
import glob
import json
import math
import re
from pathlib import Path

# [mode=eval] Step 300 [301/1], position 0,  Acceptance Rate: 0.4123
LINE = re.compile(
    r"\[mode=eval\] Step (\d+).*?position (\d+),\s*Acceptance Rate:\s*([0-9.]+)"
)
SEED_IN_PATH = re.compile(r"_s(\d+)\.log$")

# One-sided Student-t critical values at p<0.05 (no scipy dep).
T_CRIT_ONE_SIDED_05 = {1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943}


def _final_block(path):
    """Return ({position: acceptance}, last_step) for the highest-step eval block."""
    by_step = {}
    for line in Path(path).read_text(errors="ignore").splitlines():
        m = LINE.search(line)
        if not m:
            continue
        step, pos, val = int(m.group(1)), int(m.group(2)), float(m.group(3))
        by_step.setdefault(step, {})[pos] = val
    if not by_step:
        return {}, None
    last = max(by_step)
    return by_step[last], last


def final_acceptance(log_glob):
    """Glob per-seed logs, return (per_seed, last_steps) dicts keyed by seed.

    ``per_seed[seed]`` is ``{position: acceptance}`` from that run's final eval
    block; ``last_steps[seed]`` is the eval step that block came from.
    """
    paths = sorted(glob.glob(log_glob))
    per_seed = {}
    last_steps = {}
    for i, path in enumerate(paths):
        m = SEED_IN_PATH.search(path)
        seed = int(m.group(1)) if m else i
        block, last = _final_block(path)
        if not block:
            continue
        per_seed[seed] = block
        last_steps[seed] = last
    return per_seed, last_steps


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def _std(xs, ddof=1):
    n = len(xs)
    if n - ddof <= 0:
        return float("nan")
    mu = _mean(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (n - ddof))


def _paired_one_sided_t(diffs):
    """Paired one-sided t-test on diffs (H1: mean(diff) > 0).

    Returns (t, df, sig05) where ``sig05`` is True iff t exceeds the
    one-sided p<0.05 critical value at the given df.
    """
    n = len(diffs)
    if n < 2:
        return float("nan"), n - 1, False
    mu = _mean(diffs)
    sd = _std(diffs, ddof=1)
    if sd == 0 or math.isnan(sd):
        # Zero variance: significant iff strictly positive mean.
        return (float("inf") if mu > 0 else float("nan")), n - 1, mu > 0
    t = mu / (sd / math.sqrt(n))
    df = n - 1
    crit = T_CRIT_ONE_SIDED_05.get(df)
    sig = (crit is not None) and (t > crit)
    return t, df, sig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ce", required=True, help="Glob for CE per-seed logs, e.g. train_ce_s*.log"
    )
    ap.add_argument(
        "--tv", required=True, help="Glob for TV per-seed logs, e.g. train_tv_s*.log"
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ce_per_seed, ce_steps = final_acceptance(args.ce)
    tv_per_seed, tv_steps = final_acceptance(args.tv)

    positions = sorted(
        {p for d in ce_per_seed.values() for p in d}
        | {p for d in tv_per_seed.values() for p in d}
    )

    # Per-position values across seeds.
    def collect(per_seed, pos):
        return [d[pos] for d in per_seed.values() if pos in d]

    ce_by_pos = {p: collect(ce_per_seed, p) for p in positions}
    tv_by_pos = {p: collect(tv_per_seed, p) for p in positions}

    # Paired per-seed mean acceptance (mean across positions), restricted to
    # seeds present in BOTH CE and TV so the t-test is genuinely paired.
    common_seeds = sorted(set(ce_per_seed) & set(tv_per_seed))
    ce_seed_means = [_mean(list(ce_per_seed[s].values())) for s in common_seeds]
    tv_seed_means = [_mean(list(tv_per_seed[s].values())) for s in common_seeds]
    diffs = [t - c for c, t in zip(ce_seed_means, tv_seed_means)]
    t_stat, df, sig05 = _paired_one_sided_t(diffs)

    ce_grand = _mean(ce_seed_means)
    tv_grand = _mean(tv_seed_means)
    ok = bool(common_seeds) and sig05

    out = Path(args.out)
    payload = {
        "ce_logs": sorted(glob.glob(args.ce)),
        "tv_logs": sorted(glob.glob(args.tv)),
        "ce_final_steps": ce_steps,
        "tv_final_steps": tv_steps,
        "seeds": common_seeds,
        "ce_acceptance_by_pos": {p: ce_by_pos[p] for p in positions},
        "tv_acceptance_by_pos": {p: tv_by_pos[p] for p in positions},
        "ce_per_seed_mean": dict(zip(common_seeds, ce_seed_means)),
        "tv_per_seed_mean": dict(zip(common_seeds, tv_seed_means)),
        "ce_grand_mean": ce_grand,
        "tv_grand_mean": tv_grand,
        "tv_minus_ce_grand_mean": tv_grand - ce_grand,
        "paired_diffs": diffs,
        "t_statistic": t_stat,
        "df": df,
        "one_sided_p_lt_0p05": sig05,
        "claim_reproduced": ok,
    }
    (out / "acceptance.json").write_text(json.dumps(payload, indent=2, default=str))

    lines = []
    lines.append(
        "# Bebop TV-loss PoC: CE vs TV rejection-sampling acceptance (3 seeds)\n"
    )
    lines.append(
        "Qwen3-8B EAGLE3 draft head, ttt-length=3, trained on a tiny ShareGPT "
        f"slice. Acceptance = mean_v sum min(p,q) on held-out eval, at the final "
        f"eval step of each run. Seeds: {common_seeds}. "
        f"CE final steps: {ce_steps}; TV final steps: {tv_steps}.\n"
    )
    lines.append("| MTP step | CE mean ± std | TV mean ± std | TV - CE (mean) |")
    lines.append("|---|---|---|---|")
    for p in positions:
        cv, tv = ce_by_pos[p], tv_by_pos[p]
        cs = f"{_mean(cv):.4f} ± {_std(cv):.4f}" if cv else "n/a"
        ts = f"{_mean(tv):.4f} ± {_std(tv):.4f}" if tv else "n/a"
        ds = f"{_mean(tv) - _mean(cv):+.4f}" if (cv and tv) else "n/a"
        lines.append(f"| {p} | {cs} | {ts} | {ds} |")
    lines.append(
        f"| **grand** | **{ce_grand:.4f}** | **{tv_grand:.4f}** | "
        f"**{tv_grand - ce_grand:+.4f}** |"
    )
    lines.append("")
    lines.append(
        f"Paired one-sided t-test on per-seed mean acceptance (H1: TV > CE): "
        f"t = {t_stat:.3f}, df = {df}, "
        f"{'p < 0.05' if sig05 else 'p >= 0.05'} "
        f"(critical t at df={df}, one-sided 0.05 = "
        f"{T_CRIT_ONE_SIDED_05.get(df, 'n/a')}). "
        f"Per-seed paired diffs (TV - CE) = "
        f"{[f'{d:+.4f}' for d in diffs]}.\n"
    )
    verdict = "REPRODUCED" if ok else "NOT reproduced"
    lines.append(
        f"**Core claim (TV acceptance > CE acceptance, paired one-sided p<0.05): "
        f"{verdict}.** Grand-mean acceptance CE={ce_grand:.4f} vs "
        f"TV={tv_grand:.4f} (delta {tv_grand - ce_grand:+.4f}) "
        f"across {len(common_seeds)} seeds.\n"
    )
    (out / "EVAL.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
