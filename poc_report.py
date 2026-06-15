#!/usr/bin/env python3
"""Assemble the Bebop PoC CE-vs-TV comparison (EVAL.md) from eval JSON dumps.

Each fork (CE-continue, TV-finetune) writes ``eval_acceptance.json`` with the
per-MTP-step rejection-sampling acceptance (mean_v sum min(p,q)) at its final
eval. The core claim of arXiv:2606.12370 is that the TV-finetuned head reaches
HIGHER acceptance than the CE-continued head from the same warmup checkpoint.

We additionally include a compute-matched TV-from-scratch arm (no CE warmup,
trained for WARMUP+FORK steps with the TV loss) to test the paper's
justification that the TV gradient is proportional to the draft probability
q_j (Eq. 11) and therefore vanishes for a uniform/untrained head — i.e. TV is
a refinement objective. If tv_scratch underperforms CE-warm+TV-refine we
corroborate the "TV needs warmup" hypothesis; if it doesn't, the
warmup-then-refine framing is wrong.
"""
import argparse
import json
from pathlib import Path


def load(path):
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ce", required=True)
    ap.add_argument("--tv", required=True)
    ap.add_argument("--tv-scratch", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ce = load(args.ce)
    tv = load(args.tv)
    tv_scratch = load(args.tv_scratch)
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

    have_scratch = tv_scratch is not None
    if have_scratch:
        tvs_acc = tv_scratch["acceptance_by_pos"]
        n = min(n, len(tvs_acc))
        tvs_mean = sum(tvs_acc) / len(tvs_acc)
        # Paper's refinement claim is corroborated if TV-from-scratch is worse
        # than CE-warmup + TV-refine.
        refinement_claim = tvs_mean < tv_mean - 1e-4
    else:
        tvs_acc = None
        tvs_mean = None
        refinement_claim = None

    payload = {
        "ce_mean_acceptance": ce_mean,
        "tv_mean_acceptance": tv_mean,
        "tv_scratch_mean_acceptance": tvs_mean,
        "tv_minus_ce_mean": tv_mean - ce_mean,
        "tv_minus_tv_scratch_mean": (tv_mean - tvs_mean) if have_scratch else None,
        "ce_acceptance_by_pos": ce_acc,
        "tv_acceptance_by_pos": tv_acc,
        "tv_scratch_acceptance_by_pos": tvs_acc,
        "ce_eval_step": ce.get("global_step"),
        "tv_eval_step": tv.get("global_step"),
        "tv_scratch_eval_step": tv_scratch.get("global_step") if have_scratch else None,
        "claim_reproduced": bool(ok),
        "refinement_claim_corroborated": (
            bool(refinement_claim) if refinement_claim is not None else None
        ),
    }
    (out / "acceptance.json").write_text(json.dumps(payload, indent=2))

    L = []
    L.append("# Bebop TV-loss PoC: CE-continue vs TV-finetune acceptance\n")
    L.append(
        "Qwen3-8B EAGLE3 draft head, ttt-length=3. A shared CE warmup is forked "
        "into two matched-length continuations: CE-continue (standard loss) and "
        "TV-finetune (paper's TV loss). A third arm, TV-scratch, trains TV from "
        "random init for warmup+fork steps (compute-matched, no warmup ckpt) to "
        "test the paper's claim that TV is a refinement objective (Eq. 11: TV "
        "gradient ∝ q_j vanishes for a uniform head; see lk_loss.py L97). "
        "Acceptance = mean_v sum min(p,q) on held-out ShareGPT eval at the "
        "final eval step.\n"
    )
    if have_scratch:
        L.append("| MTP step | CE-continue | TV-finetune | TV-scratch | TV - CE | TV - TV-scratch |")
        L.append("|---|---|---|---|---|---|")
        for i in range(n):
            L.append(
                f"| {i} | {ce_acc[i]:.4f} | {tv_acc[i]:.4f} | {tvs_acc[i]:.4f} | "
                f"{tv_acc[i] - ce_acc[i]:+.4f} | {tv_acc[i] - tvs_acc[i]:+.4f} |"
            )
        L.append(
            f"| **mean** | **{ce_mean:.4f}** | **{tv_mean:.4f}** | **{tvs_mean:.4f}** | "
            f"**{tv_mean - ce_mean:+.4f}** | **{tv_mean - tvs_mean:+.4f}** |"
        )
    else:
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
    if have_scratch:
        ref_verdict = (
            "CORROBORATED" if refinement_claim else "NOT corroborated"
        )
        L.append(
            f"**Refinement claim (TV-scratch < CE-warm+TV-refine): {ref_verdict}.** "
            f"Mean acceptance TV-scratch={tvs_mean:.4f} vs TV-finetune={tv_mean:.4f} "
            f"(delta {tv_mean - tvs_mean:+.4f}). If TV-scratch trails TV-finetune the "
            "paper's 'TV gradient ∝ q_j vanishes for a uniform head' justification is "
            "consistent with the data; if it matches or beats TV-finetune, the "
            "warmup-then-refine framing is wrong.\n"
        )
    (out / "EVAL.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
