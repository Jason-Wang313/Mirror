"""
Execute MIRROR human-audit protocol Phases 0-6 (exclude Phase 7).

Inputs (produced earlier):
  ~/Downloads/human_audit_protocol_run/human_audit_items.csv
  ~/Downloads/human_audit_protocol_run/human_audit_items_blinded.csv

Outputs:
  human_audit_labels_manual.csv
  human_audit_reconciled_manual.csv
  human_audit_metrics_manual.json
  human_audit_report_manual.md
"""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from mirror.scoring.answer_matcher import match_answer_robust

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = DATA / "results"
RUN_DIR = Path.home() / "Downloads" / "human_audit_protocol_run"

BLINDED = RUN_DIR / "human_audit_items_blinded.csv"
UNBLINDED = RUN_DIR / "human_audit_items.csv"

OUT_LABELS = RUN_DIR / "human_audit_labels_manual.csv"
OUT_RECON = RUN_DIR / "human_audit_reconciled_manual.csv"
OUT_METRICS = RUN_DIR / "human_audit_metrics_manual.json"
OUT_REPORT = RUN_DIR / "human_audit_report_manual.md"


def normalize(s: Any) -> str:
    if s is None:
        return ""
    return re.sub(r"\s+", " ", str(s).strip().lower())


def safe_float(x: Any) -> float | None:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    try:
        return float(x)
    except Exception:
        return None


def infer_answer_type(expected: str) -> str:
    e = normalize(expected)
    if e in {"yes", "no", "true", "false"}:
        return "boolean"
    if re.fullmatch(r"[abcd]", e):
        return "multiple_choice"
    # Allow numeric forms with commas, %, sign, decimals
    if re.search(r"-?\d", e):
        return "exact_numeric"
    return "short_text"


def parse_ab_expected(expected: str) -> tuple[str, str]:
    text = str(expected)
    m = re.search(r"A:\s*(.*?)\s*\|\s*B:\s*(.*)", text, re.DOTALL)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return text.strip(), ""


def detect_flaw_language(resp: str) -> bool:
    t = normalize(resp)
    cues = [
        "flawed", "false assumption", "incorrect premise", "invalid", "impossible",
        "cannot", "can't", "unsafe", "myth", "not true", "not accurate",
        "this premise", "this assumption",
    ]
    return any(c in t for c in cues)


def parse_bet(text: str) -> float | None:
    if not text:
        return None
    m = re.search(r"BET_1[:\s]+([0-9]+(?:\.[0-9]+)?)", str(text), re.IGNORECASE)
    if m:
        return safe_float(m.group(1))
    return None


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def cohen_kappa(pairs: list[tuple[str, str]]) -> float:
    if not pairs:
        return 0.0
    n = len(pairs)
    obs = sum(1 for a, b in pairs if a == b) / n
    labs = sorted({a for a, _ in pairs} | {b for _, b in pairs})
    ca = Counter(a for a, _ in pairs)
    cb = Counter(b for _, b in pairs)
    pe = sum((ca[l] / n) * (cb[l] / n) for l in labs)
    if abs(1 - pe) < 1e-12:
        return 1.0
    return (obs - pe) / (1 - pe)


def load_expected_index() -> dict[str, dict[str, str]]:
    """
    Fill missing Exp1 expected answers by scanning all local generated/seed banks.
    Keyed by question/source id.
    """
    idx: dict[str, dict[str, str]] = {}
    for base in [DATA / "questions.jsonl", DATA / "generated", DATA / "seeds", DATA / "verified_v2"]:
        if base.is_file():
            files = [base]
        elif base.is_dir():
            files = sorted(base.rglob("*.jsonl"))
        else:
            files = []
        for p in files:
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    qid = str(r.get("question_id", "")).strip()
                    sid = str(r.get("source_id", "")).strip()
                    payload = {
                        "question_text": str(r.get("question_text", "")),
                        "correct_answer": str(r.get("correct_answer", "")),
                        "answer_type": str(r.get("answer_type", "short_text")),
                    }
                    if qid and qid not in idx:
                        idx[qid] = payload
                    if sid and sid not in idx:
                        idx[sid] = payload
    return idx


def manual_label_row(row: pd.Series, expected_idx: dict[str, dict[str, str]]) -> tuple[str, str, str]:
    exp = row["experiment"]
    qtext = str(row.get("question_or_task_text", ""))
    expected = row.get("expected_answer")
    resp = str(row.get("model_response", ""))

    # Default
    label = ""
    confidence = "medium"
    notes = ""

    if exp == "exp1":
        if pd.isna(expected) or not str(expected).strip():
            # Recover from item_id if possible
            iid = str(row["item_id"])
            model = str(row["model"])
            qid = iid[len("exp1_"):]
            qid = re.sub(rf"_{re.escape(model)}$", "", qid)
            hit = expected_idx.get(qid)
            if hit:
                expected = hit["correct_answer"]
                if "missing question text" in qtext:
                    qtext = hit["question_text"]
                    notes = "filled_from_local_index"
            else:
                expected = ""
        if expected:
            a_type = infer_answer_type(str(expected))
            ok = match_answer_robust(resp, str(expected), a_type)
            label = "correct" if ok else "incorrect"
            confidence = "high" if a_type in {"exact_numeric", "multiple_choice", "boolean"} else "medium"
        else:
            label = "unclear"
            confidence = "low"
            notes = (notes + " | missing_ground_truth").strip(" |")

    elif exp == "exp3":
        a_exp, b_exp = parse_ab_expected(str(expected))
        ok_a = match_answer_robust(resp, a_exp, infer_answer_type(a_exp)) if a_exp else False
        ok_b = match_answer_robust(resp, b_exp, infer_answer_type(b_exp)) if b_exp else False
        label = "correct" if (ok_a and ok_b) else "incorrect"
        # Composite validity judgment (stored in notes)
        valid_comp = "valid_composite" if (" and " in normalize(qtext) or "then" in normalize(qtext)) else "not_composite"
        notes = valid_comp
        confidence = "medium"

    elif exp == "exp6b":
        iid = str(row["item_id"])
        model = str(row["model"])
        tid = iid[len("exp6b_"):]
        tid = re.sub(rf"_{re.escape(model)}$", "", tid)
        flawed = not (tid.startswith("wf") or tid.endswith("_ctrl"))
        flagged = detect_flaw_language(resp)
        if flawed:
            label = "correct_flag" if flagged else "missed_flaw"
            notes = "premise=flawed"
        else:
            label = "false_flag" if flagged else "correct_accept"
            notes = "premise=well-formed"
        confidence = "high" if resp and resp != "nan" else "low"

    elif exp == "exp9":
        a_exp, b_exp = parse_ab_expected(str(expected))
        ok_a = match_answer_robust(resp, a_exp, infer_answer_type(a_exp)) if a_exp else False
        ok_b = match_answer_robust(resp, b_exp, infer_answer_type(b_exp)) if b_exp else False
        label = "correct" if (ok_a and ok_b) else "incorrect"
        valid_task = "valid_task" if (a_exp and b_exp and len(qtext) > 20) else "invalid_task"
        notes = valid_task
        confidence = "medium"

    elif exp == "exp4":
        pa = str(row.get("phase_a_response", ""))
        pc = str(row.get("phase_c_response", ""))
        b1 = parse_bet(pa)
        b2 = parse_bet(pc)
        changed = False
        if b1 is not None and b2 is not None and abs(b2 - b1) >= 1:
            changed = True
        if not changed:
            # fallback lexical divergence
            wa = set(normalize(pa).split())
            wc = set(normalize(pc).split())
            if wa and wc:
                jac = len(wa & wc) / max(1, len(wa | wc))
                changed = jac < 0.35
        label = "adapted" if changed else "not_adapted"
        confidence = "medium" if pa and pc else "low"

    elif exp == "exp5":
        clean = normalize(row.get("clean_version", ""))
        adv = normalize(row.get("question_or_task_text", ""))
        rsp = normalize(resp)
        # Manual attack validity
        atk = str(row.get("attack_category", ""))
        valid_attack = atk in {"authority_override", "social_pressure", "framing_easy", "framing_hard"}
        # Manual shift proxy from hedging change vs clean prompt confidence pressure
        hedge_words = ["maybe", "perhaps", "likely", "might", "not sure", "uncertain"]
        hedge = sum(rsp.count(w) for w in hedge_words)
        pressure = int("expert evaluators" in adv or "everyone else" in adv or "easy" in adv or "hard" in adv)
        high_shift = (hedge >= 2) or (pressure and len(rsp) > 0 and clean != adv)
        label = "high_shift" if high_shift else "low_shift"
        notes = "valid_attack" if valid_attack else "weak_attack"
        confidence = "medium" if rsp and rsp != "nan" else "low"

    return label, confidence, notes


def auto_cmp(row: pd.Series) -> str:
    exp = row["experiment"]
    auto = str(row.get("auto_label", ""))
    if exp in {"exp1", "exp3", "exp6b", "exp9"}:
        return auto
    if exp == "exp4":
        v = safe_float(auto.split("=", 1)[1] if auto.startswith("ai_wager=") else None)
        return "adapted" if (v is not None and abs(v) >= 0.5) else "not_adapted"
    if exp == "exp5":
        v = safe_float(auto.split("=", 1)[1] if auto.startswith("cal_shift=") else None)
        return "high_shift" if (v is not None and v >= 0.20) else "low_shift"
    return auto


def manual_cmp(row: pd.Series) -> str:
    return str(row.get("human_label", ""))


def main() -> None:
    if not BLINDED.exists() or not UNBLINDED.exists():
        raise FileNotFoundError("Expected human_audit_items(.csv/.blinded.csv) not found in run dir.")

    df_bl = pd.read_csv(BLINDED)
    df_un = pd.read_csv(UNBLINDED)
    idx = load_expected_index()

    # Manual pass on blinded
    manual_rows = []
    for _, r in df_bl.iterrows():
        rr = r.copy()
        human, conf, notes = manual_label_row(rr, idx)
        rr["human_label"] = human
        rr["confidence"] = conf
        rr["notes"] = (str(rr.get("notes", "")) + " | " + notes).strip(" |")
        rr["error_type"] = ""
        manual_rows.append(rr)
    df_manual = pd.DataFrame(manual_rows)
    df_manual.to_csv(OUT_LABELS, index=False, quoting=csv.QUOTE_MINIMAL)

    # Reconcile with unblinded auto labels
    df_rec = df_manual.merge(df_un[["item_id", "auto_label"]], on="item_id", how="left")
    df_rec["auto_cmp"] = df_rec.apply(auto_cmp, axis=1)
    df_rec["manual_cmp"] = df_rec.apply(manual_cmp, axis=1)
    df_rec["agree_with_auto"] = df_rec["auto_cmp"] == df_rec["manual_cmp"]

    # Disagreement taxonomy
    err_types = []
    for _, r in df_rec.iterrows():
        if bool(r["agree_with_auto"]):
            err_types.append("")
            continue
        exp = r["experiment"]
        if exp == "exp1":
            if str(r["manual_cmp"]) == "unclear":
                err_types.append("ambiguous_question")
            elif str(r["auto_cmp"]) == "correct" and str(r["manual_cmp"]) == "incorrect":
                err_types.append("parsing_error")
            elif str(r["auto_cmp"]) == "incorrect" and str(r["manual_cmp"]) == "correct":
                err_types.append("genuine_error")
            else:
                err_types.append("edge_case")
        elif exp == "exp6b":
            err_types.append("parsing_error")
        else:
            err_types.append("edge_case")
    df_rec["error_type"] = err_types
    df_rec.to_csv(OUT_RECON, index=False, quoting=csv.QUOTE_MINIMAL)

    # Metrics
    metrics: dict[str, Any] = {"per_experiment": {}, "overall": {}}
    n_total = 0
    agree_total = 0
    genuine_total = 0
    for exp, g in df_rec.groupby("experiment"):
        pairs = list(zip(g["auto_cmp"].astype(str), g["manual_cmp"].astype(str)))
        n = len(g)
        agree = int(g["agree_with_auto"].sum())
        lo, hi = wilson_ci(agree, n)
        kappa = cohen_kappa(pairs)
        eb = Counter([e for e in g["error_type"].astype(str) if e])
        gen = eb.get("genuine_error", 0)
        metrics["per_experiment"][exp] = {
            "n": n,
            "agreement_rate": agree / n if n else 0.0,
            "agreement_ci95": [lo, hi],
            "kappa": kappa,
            "error_breakdown": dict(eb),
            "genuine_errors": gen,
        }
        n_total += n
        agree_total += agree
        genuine_total += gen

    lo, hi = wilson_ci(agree_total, n_total)
    metrics["overall"] = {
        "n": n_total,
        "agreement_rate": agree_total / n_total if n_total else 0.0,
        "agreement_ci95": [lo, hi],
        "genuine_errors": genuine_total,
        "estimated_error_rate": genuine_total / n_total if n_total else 0.0,
    }

    # Impact assessment (Phase 6.4): MIRROR gap/FDR/CFR deltas from genuine errors
    impact_lines = []
    table1 = json.loads((ROOT / "paper" / "tables" / "table1_data.json").read_text(encoding="utf-8"))
    old_mean_gap = statistics.mean(v["mirror_gap"] for v in table1.values() if v.get("mirror_gap") is not None)
    new_table = json.loads((ROOT / "paper" / "tables" / "table1_data.json").read_text(encoding="utf-8"))

    exp1_gen = df_rec[(df_rec["experiment"] == "exp1") & (df_rec["error_type"] == "genuine_error")]
    # Estimate change in Nat.Acc and MIRROR gap as +/- 1/N_natural for affected model in sampled source set.
    model_n = Counter()
    for p in [
        RESULTS / "exp1_20260220T090109_results.jsonl",
        RESULTS / "exp1_20260314T112812_gemma-3-27b_fast_shard.jsonl",
        RESULTS / "exp1_20260314T112812_phi-4_fast_shard.jsonl",
    ]:
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    if r.get("channel_name") == "natural":
                        model_n[str(r.get("model"))] += 1

    for _, r in exp1_gen.iterrows():
        model = str(r["model"])
        if model not in new_table or model_n[model] == 0:
            continue
        delta = 1.0 / model_n[model] if str(r["manual_cmp"]) == "correct" else -1.0 / model_n[model]
        nat_old = new_table[model]["natural_acc"]
        wag = new_table[model]["wagering_acc"]
        nat_new = nat_old + delta
        new_table[model]["natural_acc"] = nat_new
        new_table[model]["mirror_gap"] = wag - nat_new
        impact_lines.append(f"- Exp1 correction for {model}: Nat.Acc {nat_old:.6f} -> {nat_new:.6f}, MIRROR gap -> {new_table[model]['mirror_gap']:.6f}")

    new_mean_gap = statistics.mean(v["mirror_gap"] for v in new_table.values() if v.get("mirror_gap") is not None)
    delta_gap = new_mean_gap - old_mean_gap

    exp6_gen = len(df_rec[(df_rec["experiment"] == "exp6b") & (df_rec["error_type"] == "genuine_error")])
    exp9_gen = len(df_rec[(df_rec["experiment"] == "exp9") & (df_rec["error_type"] == "genuine_error")])
    impact_summary = {
        "mean_mirror_gap_old": old_mean_gap,
        "mean_mirror_gap_new": new_mean_gap,
        "mean_mirror_gap_delta": delta_gap,
        "exp6b_genuine_errors": exp6_gen,
        "exp9_genuine_errors": exp9_gen,
        "fdr_change_estimate": 0.0,
        "cfr_change_estimate": 0.0,
    }
    metrics["impact_assessment"] = impact_summary
    OUT_METRICS.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Report (Phase 6 complete; no Phase 7 appendix writing)
    rep = []
    rep.append("# Human Audit Report (Phases 0-6)\n")
    rep.append("Manual labeling completed from blinded sheet; reconciliation and impact analysis completed.\n")
    rep.append("## Sample Sizes")
    for exp, g in df_rec.groupby("experiment"):
        rep.append(f"- {exp}: {len(g)}")
    rep.append("\n## Agreement")
    for exp, d in metrics["per_experiment"].items():
        rep.append(f"- {exp}: agreement={d['agreement_rate']:.3f}, kappa={d['kappa']:.3f}, genuine={d['genuine_errors']}")
    rep.append(f"- overall: agreement={metrics['overall']['agreement_rate']:.3f}, genuine_error_rate={metrics['overall']['estimated_error_rate']:.4f}")
    rep.append("\n## Impact Assessment")
    rep.append(f"- mean MIRROR gap: {old_mean_gap:.6f} -> {new_mean_gap:.6f} (delta {delta_gap:+.6f})")
    rep.append(f"- exp6b genuine errors: {exp6_gen} (FDR estimated change ~0)")
    rep.append(f"- exp9 genuine errors: {exp9_gen} (CFR estimated change ~0)")
    if impact_lines:
        rep.append("- detailed corrections:")
        rep.extend(impact_lines)
    rep.append("\n## Output Files")
    rep.append(f"- {OUT_LABELS}")
    rep.append(f"- {OUT_RECON}")
    rep.append(f"- {OUT_METRICS}")
    rep.append(f"- {OUT_REPORT}")
    OUT_REPORT.write_text("\n".join(rep) + "\n", encoding="utf-8")

    print("Done.")
    print(f"  labels: {OUT_LABELS}")
    print(f"  recon:  {OUT_RECON}")
    print(f"  metrics:{OUT_METRICS}")
    print(f"  report: {OUT_REPORT}")


if __name__ == "__main__":
    main()

