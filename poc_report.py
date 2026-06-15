#!/usr/bin/env python3
"""Assemble the Bebop PoC CE-vs-TV comparison (EVAL.md) from eval JSON dumps.

Each fork (CE-continue, TV-finetune) writes ``eval_acceptance.json`` with the
per-MTP-step rejection-sampling acceptance (mean_v sum min(p,q)) at its final
eval. The core claim of arXiv:2606.12370 is that the TV-finetuned head reaches
HIGHER acceptance than the CE-continued head from the same warmup checkpoint.
"""
import argparse
import json
from pathlib import Path


def load(path):
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ce", required=True)
    ap.add_argument("--tv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ce = load(args.ce)
    tv = load(args.tv)
    out = Path(args.out)

    if ce is None or tv is None:
        msg = f"# Bebop TV-loss PoC: FAILED\n\nMissing eval JSON (ce={ce is not None}, tv={tv is not None}).\n"
        (out / "EVAL.md").write_text(msg)
        print(msg)
        raise SystemExit(1)

    ce_acc = ce["acceptance_by_pos"]
    tv_acc = tv["acceptance_by_pos"]
    n = min(len(ce_acc), len(tv_acc))
    ce_mean = sum(ce_acc) / len(ce_acc)
    tv_mean = sum(tv_acc) / len(tv_acc)
    ok = tv_mean >= ce_mean - 1e-4

    payload = {
        "ce_mean_acceptance": ce_mean,
        "tv_mean_acceptance": tv_mean,
        "tv_minus_ce_mean": tv_mean - ce_mean,
        "ce_acceptance_by_pos": ce_acc,
        "tv_acceptance_by_pos": tv_acc,
        "ce_eval_step": ce.get("global_step"),
        "tv_eval_step": tv.get("global_step"),
        "claim_reproduced": bool(ok),
    }
    (out / "acceptance.json").write_text(json.dumps(payload, indent=2))

    L = []
    L.append("# Bebop TV-loss PoC: CE-continue vs TV-finetune acceptance\n")
    L.append(
        "Qwen3-8B EAGLE3 draft head, ttt-length=3. A shared CE warmup is forked "
        "into two matched-length continuations: CE-continue (standard loss) and "
        "TV-finetune (paper's TV loss). Acceptance = mean_v sum min(p,q) on "
        "held-out ShareGPT eval at the final eval step.\n"
    )
    L.append("| MTP step | CE-continue | TV-finetune | TV - CE |")
    L.append("|---|---|---|---|")
    for i in range(n):
        L.append(
            f"| {i} | {ce_acc[i]:.4f} | {tv_acc[i]:.4f} | {tv_acc[i] - ce_acc[i]:+.4f} |"
        )
    L.append(
        f"| **mean** | **{ce_mean:.4f}** | **{tv_mean:.4f}** | **{tv_mean - ce_mean:+.4f}** |"
    )
    L.append("")
    verdict = "REPRODUCED" if ok else "NOT reproduced"
    L.append(
        f"**Core claim (TV acceptance >= CE acceptance): {verdict}.** "
        f"Mean acceptance CE={ce_mean:.4f} vs TV={tv_mean:.4f} "
        f"(delta {tv_mean - ce_mean:+.4f}).\n"
    )
    (out / "EVAL.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
