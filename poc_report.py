#!/usr/bin/env python3
"""Parse CE vs TV training logs and emit the Bebop PoC comparison (EVAL.md).

Reads the per-MTP-step rejection-sampling acceptance rates that
``train_eagle3.py`` prints at eval time (lines tagged ``[mode=eval]``),
keeps the final eval block from each run, and writes a CE-vs-TV table plus a
machine-readable JSON. The core claim of arXiv:2606.12370 is that the
TV-trained head reaches >= CE acceptance at matched training.
"""
import argparse
import json
import re
from pathlib import Path

# [mode=eval] Step 300 [301/1], position 0,  Acceptance Rate: 0.4123
LINE = re.compile(
    r"\[mode=eval\] Step (\d+).*?position (\d+),\s*Acceptance Rate:\s*([0-9.]+)"
)


def final_acceptance(log_path):
    """Return {position: acceptance} from the highest-step eval block."""
    by_step = {}
    for line in Path(log_path).read_text(errors="ignore").splitlines():
        m = LINE.search(line)
        if not m:
            continue
        step, pos, val = int(m.group(1)), int(m.group(2)), float(m.group(3))
        by_step.setdefault(step, {})[pos] = val
    if not by_step:
        return {}, None
    last = max(by_step)
    return by_step[last], last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ce", required=True)
    ap.add_argument("--tv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ce, ce_step = final_acceptance(args.ce)
    tv, tv_step = final_acceptance(args.tv)
    positions = sorted(set(ce) | set(tv))

    def mean(d):
        return sum(d.values()) / len(d) if d else float("nan")

    def gamma(d):
        """Expected accepted-prefix length γ = Σ_{k=1..K} Π_{i<k} α_i.

        With α_i = per-position acceptance rate (positions sorted ascending)
        and K = number of MTP positions, this is the chain-acceptance length
        that controls speculative-decoding wall-clock speedup:
            γ = 1 + α_0 + α_0·α_1 + ... + α_0·α_1·...·α_{K-2}   (K terms).
        """
        if not d:
            return float("nan")
        alphas = [d[p] for p in sorted(d)]
        g, prod = 0.0, 1.0
        for a in alphas:
            g += prod  # adds Π_{i<k} α_i for k = 1, 2, ..., K
            prod *= a
        return g

    ce_mean, tv_mean = mean(ce), mean(tv)
    ce_gamma, tv_gamma = gamma(ce), gamma(tv)
    K = len(positions) if positions else 0
    ce_gamma_norm = ce_gamma / K if K else float("nan")
    tv_gamma_norm = tv_gamma / K if K else float("nan")
    ok = bool(ce and tv) and tv_mean >= ce_mean - 1e-4

    out = Path(args.out)
    payload = {
        "ce_step": ce_step,
        "tv_step": tv_step,
        "ce_acceptance_by_pos": ce,
        "tv_acceptance_by_pos": tv,
        "ce_mean_acceptance": ce_mean,
        "tv_mean_acceptance": tv_mean,
        "tv_minus_ce_mean": tv_mean - ce_mean,
        "K": K,
        "ce_gamma": ce_gamma,
        "tv_gamma": tv_gamma,
        "tv_minus_ce_gamma": tv_gamma - ce_gamma,
        "ce_gamma_per_step": ce_gamma_norm,
        "tv_gamma_per_step": tv_gamma_norm,
        "claim_reproduced": ok,
    }
    (out / "acceptance.json").write_text(json.dumps(payload, indent=2))

    lines = []
    lines.append("# Bebop TV-loss PoC: CE vs TV rejection-sampling acceptance\n")
    lines.append(
        "Qwen3-8B EAGLE3 draft head, ttt-length=3, trained on a tiny ShareGPT "
        f"slice. Acceptance = mean_v sum min(p,q) on held-out eval, at the final "
        f"eval step (CE step {ce_step}, TV step {tv_step}).\n"
    )
    lines.append("| MTP step | CE acceptance | TV acceptance | TV - CE |")
    lines.append("|---|---|---|---|")
    for p in positions:
        c = ce.get(p)
        t = tv.get(p)
        cs = f"{c:.4f}" if c is not None else "n/a"
        ts = f"{t:.4f}" if t is not None else "n/a"
        ds = f"{t - c:+.4f}" if (c is not None and t is not None) else "n/a"
        lines.append(f"| {p} | {cs} | {ts} | {ds} |")
    lines.append(
        f"| **mean** | **{ce_mean:.4f}** | **{tv_mean:.4f}** | **{tv_mean - ce_mean:+.4f}** |"
    )
    lines.append(
        f"| **γ = Σ_k Π_{{i<k}} α_i** | **{ce_gamma:.4f}** | **{tv_gamma:.4f}** | "
        f"**{tv_gamma - ce_gamma:+.4f}** |"
    )
    lines.append(
        f"| **γ / K** (K={K}) | **{ce_gamma_norm:.4f}** | **{tv_gamma_norm:.4f}** | "
        f"**{tv_gamma_norm - ce_gamma_norm:+.4f}** |"
    )
    lines.append("")
    verdict = "REPRODUCED" if ok else "NOT reproduced"
    lines.append(
        f"**Core claim (TV acceptance >= CE acceptance): {verdict}.** "
        f"Mean acceptance CE={ce_mean:.4f} vs TV={tv_mean:.4f} "
        f"(delta {tv_mean - ce_mean:+.4f}).\n"
    )
    lines.append(
        f"Expected accepted-prefix length γ (the speedup proxy): "
        f"CE={ce_gamma:.4f} vs TV={tv_gamma:.4f} (delta {tv_gamma - ce_gamma:+.4f}); "
        f"γ/K: CE={ce_gamma_norm:.4f} vs TV={tv_gamma_norm:.4f} "
        f"(delta {tv_gamma_norm - ce_gamma_norm:+.4f}).\n"
    )
    (out / "EVAL.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
