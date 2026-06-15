#!/usr/bin/env python3
"""Parse CE vs TV vs alpha training logs and emit the Bebop PoC report.

Reads the per-MTP-step rejection-sampling acceptance rates that
``train_eagle3.py`` prints at eval time (lines tagged ``[mode=eval]``),
keeps the final eval block from each run, and writes a CE-vs-TV-vs-alpha
table plus a machine-readable JSON. The core claim of arXiv:2606.12370 is
that an LK-style objective (TV: 1 - alpha, or alpha: -log alpha) reaches
>= CE acceptance at matched training; the alpha arm isolates whether the
log-margin variant dominates raw TV.
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
    ap.add_argument("--alpha", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ce, ce_step = final_acceptance(args.ce)
    tv, tv_step = final_acceptance(args.tv)
    al, al_step = final_acceptance(args.alpha)
    positions = sorted(set(ce) | set(tv) | set(al))

    def mean(d):
        return sum(d.values()) / len(d) if d else float("nan")

    ce_mean, tv_mean, al_mean = mean(ce), mean(tv), mean(al)
    tv_ok = bool(ce and tv) and tv_mean >= ce_mean - 1e-4
    alpha_ok = bool(ce and al) and al_mean >= ce_mean - 1e-4
    alpha_beats_tv = bool(tv and al) and al_mean > tv_mean + 1e-4

    out = Path(args.out)
    payload = {
        "ce_step": ce_step,
        "tv_step": tv_step,
        "alpha_step": al_step,
        "ce_acceptance_by_pos": ce,
        "tv_acceptance_by_pos": tv,
        "alpha_acceptance_by_pos": al,
        "ce_mean_acceptance": ce_mean,
        "tv_mean_acceptance": tv_mean,
        "alpha_mean_acceptance": al_mean,
        "tv_minus_ce_mean": tv_mean - ce_mean,
        "alpha_minus_ce_mean": al_mean - ce_mean,
        "alpha_minus_tv_mean": al_mean - tv_mean,
        "tv_claim_reproduced": tv_ok,
        "alpha_claim_reproduced": alpha_ok,
        "alpha_beats_tv": alpha_beats_tv,
    }
    (out / "acceptance.json").write_text(json.dumps(payload, indent=2))

    lines = []
    lines.append(
        "# Bebop LK-loss PoC: CE vs TV vs alpha rejection-sampling acceptance\n"
    )
    lines.append(
        "Qwen3-8B EAGLE3 draft head, ttt-length=3, trained on a tiny ShareGPT "
        "slice. Acceptance = mean_v sum min(p,q) on held-out eval, at the final "
        f"eval step (CE step {ce_step}, TV step {tv_step}, alpha step {al_step}). "
        "TV minimizes (1 - alpha); the alpha arm minimizes (-log alpha), which "
        "applies stronger gradients near alpha~=1.\n"
    )
    lines.append(
        "| MTP step | CE acceptance | TV acceptance | alpha acceptance "
        "| TV - CE | alpha - CE | alpha - TV |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for p in positions:
        c = ce.get(p)
        t = tv.get(p)
        a = al.get(p)
        cs = f"{c:.4f}" if c is not None else "n/a"
        ts = f"{t:.4f}" if t is not None else "n/a"
        as_ = f"{a:.4f}" if a is not None else "n/a"
        d_tc = f"{t - c:+.4f}" if (c is not None and t is not None) else "n/a"
        d_ac = f"{a - c:+.4f}" if (c is not None and a is not None) else "n/a"
        d_at = f"{a - t:+.4f}" if (t is not None and a is not None) else "n/a"
        lines.append(f"| {p} | {cs} | {ts} | {as_} | {d_tc} | {d_ac} | {d_at} |")
    lines.append(
        f"| **mean** | **{ce_mean:.4f}** | **{tv_mean:.4f}** | **{al_mean:.4f}** "
        f"| **{tv_mean - ce_mean:+.4f}** | **{al_mean - ce_mean:+.4f}** "
        f"| **{al_mean - tv_mean:+.4f}** |"
    )
    lines.append("")
    tv_verdict = "REPRODUCED" if tv_ok else "NOT reproduced"
    alpha_verdict = "REPRODUCED" if alpha_ok else "NOT reproduced"
    winner = (
        "alpha" if (al_mean > tv_mean and al_mean > ce_mean)
        else "TV" if (tv_mean > ce_mean) else "CE"
    )
    lines.append(
        f"**TV vs CE (TV acceptance >= CE acceptance): {tv_verdict}.** "
        f"Mean acceptance CE={ce_mean:.4f} vs TV={tv_mean:.4f} "
        f"(delta {tv_mean - ce_mean:+.4f}).\n"
    )
    lines.append(
        f"**alpha vs CE (alpha acceptance >= CE acceptance): {alpha_verdict}.** "
        f"Mean acceptance CE={ce_mean:.4f} vs alpha={al_mean:.4f} "
        f"(delta {al_mean - ce_mean:+.4f}).\n"
    )
    lines.append(
        f"**alpha vs TV (does -log alpha dominate plain TV?): "
        f"{'YES' if alpha_beats_tv else 'NO'}** "
        f"(delta alpha - TV = {al_mean - tv_mean:+.4f}). "
        f"Best LK objective at matched training: **{winner}**.\n"
    )
    (out / "EVAL.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
