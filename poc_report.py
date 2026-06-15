#!/usr/bin/env python3
"""Parse CE / TV / lambda training logs and emit the Bebop PoC comparison (EVAL.md).

Reads the per-MTP-step rejection-sampling acceptance rates that
``train_eagle3.py`` prints at eval time (lines tagged ``[mode=eval]``),
keeps the final eval block from each run, and writes a CE-vs-TV-vs-lambda
table plus a machine-readable JSON. The core claim of arXiv:2606.12370 is
that the TV-trained head reaches >= CE acceptance at matched training; the
lambda arm tests whether an adaptive KL+TV hybrid beats pure TV.
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
    """Return ({position: acceptance}, step) from the highest-step eval block."""
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
    ap.add_argument("--lambda-log", dest="lam", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ce, ce_step = final_acceptance(args.ce)
    tv, tv_step = final_acceptance(args.tv)
    lam, lam_step = final_acceptance(args.lam)
    positions = sorted(set(ce) | set(tv) | set(lam))

    def mean(d):
        return sum(d.values()) / len(d) if d else float("nan")

    ce_mean, tv_mean, lam_mean = mean(ce), mean(tv), mean(lam)
    tv_ok = bool(ce and tv) and tv_mean >= ce_mean - 1e-4
    lam_beats_tv = bool(tv and lam) and lam_mean > tv_mean + 1e-4

    out = Path(args.out)
    payload = {
        "ce_step": ce_step,
        "tv_step": tv_step,
        "lambda_step": lam_step,
        "ce_acceptance_by_pos": ce,
        "tv_acceptance_by_pos": tv,
        "lambda_acceptance_by_pos": lam,
        "ce_mean_acceptance": ce_mean,
        "tv_mean_acceptance": tv_mean,
        "lambda_mean_acceptance": lam_mean,
        "tv_minus_ce_mean": tv_mean - ce_mean,
        "lambda_minus_ce_mean": lam_mean - ce_mean,
        "lambda_minus_tv_mean": lam_mean - tv_mean,
        "claim_reproduced": tv_ok,
        "lambda_beats_tv": lam_beats_tv,
    }
    (out / "acceptance.json").write_text(json.dumps(payload, indent=2))

    lines = []
    lines.append(
        "# Bebop TV-loss PoC: CE vs TV vs lambda (KL+TV) rejection-sampling "
        "acceptance\n"
    )
    lines.append(
        "Qwen3-8B EAGLE3 draft head, ttt-length=3, trained on a tiny ShareGPT "
        f"slice. Acceptance = mean_v sum min(p,q) on held-out eval, at the final "
        f"eval step (CE step {ce_step}, TV step {tv_step}, lambda step {lam_step}). "
        "The lambda arm uses the adaptive hybrid "
        "kl_weight*KL + (1-kl_weight)*(1-alpha) with "
        "kl_weight = kl_scale*exp(-kl_decay*alpha), kl_scale=1.0, kl_decay=3.0.\n"
    )
    lines.append(
        "| MTP step | CE acceptance | TV acceptance | lambda acceptance | "
        "TV - CE | lambda - CE | lambda - TV |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for p in positions:
        c = ce.get(p)
        t = tv.get(p)
        l = lam.get(p)
        cs = f"{c:.4f}" if c is not None else "n/a"
        ts = f"{t:.4f}" if t is not None else "n/a"
        ls = f"{l:.4f}" if l is not None else "n/a"
        tc = f"{t - c:+.4f}" if (c is not None and t is not None) else "n/a"
        lc = f"{l - c:+.4f}" if (c is not None and l is not None) else "n/a"
        lt = f"{l - t:+.4f}" if (t is not None and l is not None) else "n/a"
        lines.append(f"| {p} | {cs} | {ts} | {ls} | {tc} | {lc} | {lt} |")
    lines.append(
        f"| **mean** | **{ce_mean:.4f}** | **{tv_mean:.4f}** | **{lam_mean:.4f}** "
        f"| **{tv_mean - ce_mean:+.4f}** | **{lam_mean - ce_mean:+.4f}** "
        f"| **{lam_mean - tv_mean:+.4f}** |"
    )
    lines.append("")
    verdict = "REPRODUCED" if tv_ok else "NOT reproduced"
    lines.append(
        f"**Core claim (TV acceptance >= CE acceptance): {verdict}.** "
        f"Mean acceptance CE={ce_mean:.4f} vs TV={tv_mean:.4f} "
        f"(delta {tv_mean - ce_mean:+.4f}).\n"
    )
    lam_verdict = "YES" if lam_beats_tv else "no"
    lines.append(
        f"**Adaptive lambda (KL+TV) beats pure TV: {lam_verdict}.** "
        f"Mean acceptance TV={tv_mean:.4f} vs lambda={lam_mean:.4f} "
        f"(delta {lam_mean - tv_mean:+.4f}).\n"
    )
    (out / "EVAL.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
