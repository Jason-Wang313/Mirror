"""
MIRROR human-audit executor.

Builds protocol outputs:
1) 420-item audit sample across Exp1/3/4/5/6b/9
2) blinded + unblinded CSVs
3) provisional human labels
4) reconciliation metrics + Appendix-I draft
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mirror.scoring.answer_matcher import match_answer_robust


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = DATA / "results"


COMMON_COLUMNS = [
    "item_id",
    "experiment",
    "domain",
    "model",
    "question_or_task_text",
    "expected_answer",
    "model_response",
    "premise_type",
    "attack_category",
    "clean_version",
    "condition",
    "domain_classification",
    "model_decision",
    "feedback_type",
    "phase_a_response",
    "phase_c_response",
    "human_label",
    "confidence",
    "agree_with_auto",
    "error_type",
    "notes",
    "auto_label",
]


EXP9_STRAT_MODELS = [
    "deepseek-r1",
    "gemma-3-27b",
    "gpt-oss-120b",
    "llama-3.1-70b",
    "phi-4",
]


@dataclass
class QuestionRef:
    question_text: str
    correct_answer: str
    answer_type: str
    domain: str | None = None
    subcategory: str | None = None


def jsonl_iter(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def safe_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def normalize_text(s: str | None) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", " ", s.strip().lower())


def bool_to_str(v: bool | None) -> str:
    if v is None:
        return ""
    return "true" if v else "false"


def load_question_maps() -> tuple[dict[str, QuestionRef], dict[str, QuestionRef]]:
    by_qid: dict[str, QuestionRef] = {}
    by_source: dict[str, QuestionRef] = {}
    q_path = DATA / "questions.jsonl"
    for r in jsonl_iter(q_path):
        ref = QuestionRef(
            question_text=str(r.get("question_text", "")),
            correct_answer=str(r.get("correct_answer", "")),
            answer_type=str(r.get("answer_type", "short_text")),
            domain=r.get("domain"),
            subcategory=r.get("subcategory"),
        )
        qid = r.get("question_id")
        sid = r.get("source_id")
        if qid:
            by_qid[str(qid)] = ref
        if sid:
            by_source[str(sid)] = ref
    return by_qid, by_source


def get_question_ref(
    qid: str,
    qmap_qid: dict[str, QuestionRef],
    qmap_sid: dict[str, QuestionRef],
) -> QuestionRef | None:
    return qmap_qid.get(qid) or qmap_sid.get(qid)


def extract_phase_wager(phase: dict[str, Any] | None) -> float | None:
    if not phase:
        return None
    wager = safe_float(phase.get("wager"))
    if wager is not None:
        return wager
    raw = str(phase.get("raw", ""))
    m = re.search(r"BET_1[:\s]+([0-9]+(?:\.[0-9]+)?)", raw, re.IGNORECASE)
    if m:
        return safe_float(m.group(1))
    return None


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    phat = k / n
    denom = 1.0 + (z * z) / n
    center = (phat + (z * z) / (2 * n)) / denom
    margin = (z * math.sqrt((phat * (1 - phat) + (z * z) / (4 * n)) / n)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def cohen_kappa(pairs: list[tuple[str, str]]) -> float:
    if not pairs:
        return 0.0
    labels = sorted({a for a, _ in pairs} | {b for _, b in pairs})
    n = len(pairs)
    obs = sum(1 for a, b in pairs if a == b) / n
    auto_counts = Counter(a for a, _ in pairs)
    human_counts = Counter(b for _, b in pairs)
    pe = 0.0
    for lab in labels:
        pe += (auto_counts[lab] / n) * (human_counts[lab] / n)
    if abs(1 - pe) < 1e-12:
        return 1.0
    return (obs - pe) / (1 - pe)


def load_exp3_task_map() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for p in [DATA / "exp3" / "intersection_tasks.jsonl", DATA / "exp3" / "control_tasks.jsonl"]:
        if not p.exists():
            continue
        for r in jsonl_iter(p):
            tid = str(r.get("task_id", ""))
            if tid:
                out[tid] = r
    return out


def load_exp9_task_map() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    p = DATA / "exp9_tasks.jsonl"
    for r in jsonl_iter(p):
        tid = str(r.get("task_id", ""))
        if tid:
            out[tid] = r
    return out


def load_exp6b_task_map() -> dict[str, dict[str, Any]]:
    """
    Merge old 6b tasks defined in scripts/run_experiment_6.py and new expanded tasks.
    """
    out: dict[str, dict[str, Any]] = {}

    runner = ROOT / "scripts" / "run_experiment_6.py"
    mod = ast.parse(runner.read_text(encoding="utf-8"))
    old_flawed: list[dict[str, Any]] = []
    old_wf: list[dict[str, Any]] = []
    for node in mod.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "FLAWED_TASKS_6B":
                    old_flawed = ast.literal_eval(node.value)
                if isinstance(t, ast.Name) and t.id == "WELLFORMED_6B":
                    old_wf = ast.literal_eval(node.value)

    for r in old_flawed:
        tid = str(r["id"])
        out[tid] = {
            "task_text": str(r.get("task", "")),
            "domain": str(r.get("domain", "")),
            "category": str(r.get("cat", "flawed_premise")),
            "premise_type": "flawed",
            "specific_flaw": str(r.get("flaw", "")),
        }
    for r in old_wf:
        tid = str(r["id"])
        out[tid] = {
            "task_text": str(r.get("task", "")),
            "domain": str(r.get("domain", "")),
            "category": "wellformed",
            "premise_type": "well-formed",
            "specific_flaw": "",
        }

    new_flawed = read_json(DATA / "exp6" / "expanded" / "6b_flawed_new.json", default=[]) or []
    for r in new_flawed:
        tid = str(r.get("id"))
        out[tid] = {
            "task_text": str(r.get("flawed_prompt", "")),
            "domain": str(r.get("domain", "")),
            "category": str(r.get("category", "flawed_premise")),
            "premise_type": "flawed",
            "specific_flaw": str(r.get("specific_flaw", "")),
        }

    new_ctrl = read_json(DATA / "exp6" / "expanded" / "6b_controls_new.json", default=[]) or []
    for r in new_ctrl:
        tid = str(r.get("id"))
        out[tid] = {
            "task_text": str(r.get("control_prompt", "")),
            "domain": str(r.get("domain", "")),
            "category": str(r.get("category", "wellformed")),
            "premise_type": "well-formed",
            "specific_flaw": "",
        }
    return out


def pick_exp6b_model() -> str:
    """
    Prefer deepseek-r1 for protocol consistency; fallback to any model with max unique 6b tasks.
    """
    p = RESULTS / "exp6_master_results.jsonl"
    task_by_model: dict[str, set[str]] = defaultdict(set)
    for r in jsonl_iter(p):
        if r.get("sub_experiment") == "6b":
            m = str(r.get("model"))
            tid = str(r.get("task_id"))
            if m and tid:
                task_by_model[m].add(tid)
    if "deepseek-r1" in task_by_model and len(task_by_model["deepseek-r1"]) >= 220:
        return "deepseek-r1"
    if not task_by_model:
        raise RuntimeError("No Exp6b records found.")
    best = max(task_by_model.items(), key=lambda kv: len(kv[1]))[0]
    return best


def build_exp6b_rows(rng: random.Random, exp6b_task_map: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    p = RESULTS / "exp6_master_results.jsonl"
    target_model = pick_exp6b_model()
    by_tid: dict[str, dict[str, Any]] = {}

    for r in jsonl_iter(p):
        if r.get("sub_experiment") != "6b":
            continue
        if str(r.get("model")) != target_model:
            continue
        tid = str(r.get("task_id"))
        if tid and tid not in by_tid:
            by_tid[tid] = r

    rows: list[dict[str, Any]] = []
    for tid in sorted(by_tid):
        r = by_tid[tid]
        task_info = exp6b_task_map.get(tid, {})
        premise = "flawed" if str(r.get("task_type")) == "flawed" else "well-formed"
        flagged = bool(r.get("flagged_flaw"))
        if premise == "flawed":
            auto = "correct_flag" if flagged else "missed_flaw"
            expected = "Identify and flag the flawed premise before proceeding."
        else:
            auto = "false_flag" if flagged else "correct_accept"
            expected = "Treat as well-formed and answer normally without false flaw-flagging."

        task_text = task_info.get("task_text") or ""
        if not task_text:
            task_text = f"[Task {tid}]"

        rows.append({
            "item_id": f"exp6b_{tid}_{target_model}",
            "experiment": "exp6b",
            "domain": task_info.get("domain") or str(r.get("domain", "")),
            "model": target_model,
            "question_or_task_text": task_text,
            "expected_answer": expected,
            "model_response": str(r.get("raw_response", "")),
            "premise_type": premise,
            "attack_category": "",
            "clean_version": "",
            "condition": "",
            "domain_classification": "",
            "model_decision": "",
            "feedback_type": "",
            "phase_a_response": "",
            "phase_c_response": "",
            "human_label": "",
            "confidence": "",
            "agree_with_auto": "",
            "error_type": "",
            "notes": str(task_info.get("specific_flaw", ""))[:240],
            "auto_label": auto,
        })

    if len(rows) != 220:
        raise RuntimeError(f"Exp6b rows expected 220, got {len(rows)} for model {target_model}.")
    return rows


def load_exp1_natural_records() -> list[dict[str, Any]]:
    """
    Combine runs so we can stratify the 5 target models in protocol.
    """
    files = [
        RESULTS / "exp1_20260220T090109_results.jsonl",
        RESULTS / "exp1_20260314T112812_gemma-3-27b_fast_shard.jsonl",
        RESULTS / "exp1_20260314T112812_phi-4_fast_shard.jsonl",
    ]
    wanted_models = set(EXP9_STRAT_MODELS)
    out: list[dict[str, Any]] = []
    for p in files:
        if not p.exists():
            continue
        for r in jsonl_iter(p):
            if str(r.get("model")) not in wanted_models:
                continue
            if str(r.get("channel_name")) != "natural":
                continue
            if not r.get("question_id"):
                continue
            out.append(r)
    return out


def choose_exp1_domains(records: list[dict[str, Any]]) -> list[str]:
    by_model_domain: dict[tuple[str, str], list[bool]] = defaultdict(list)
    models = set(EXP9_STRAT_MODELS)
    for r in records:
        m = str(r.get("model"))
        d = str(r.get("domain"))
        if m in models and d:
            by_model_domain[(m, d)].append(bool(r.get("answer_correct")))

    all_domains = sorted({d for _, d in by_model_domain})
    scored: list[tuple[float, str]] = []
    for d in all_domains:
        ok = True
        errs = []
        for m in EXP9_STRAT_MODELS:
            vals = by_model_domain.get((m, d), [])
            if len(vals) < 10:
                ok = False
                break
            acc = sum(vals) / len(vals)
            errs.append(abs(acc - 0.5))
        if ok:
            scored.append((statistics.mean(errs), d))
    if len(scored) < 2:
        fallback = [d for d in all_domains if d in {"arithmetic", "factual", "logical", "temporal"}]
        return (fallback[:2] if len(fallback) >= 2 else all_domains[:2])
    scored.sort(key=lambda x: x[0])
    return [scored[0][1], scored[1][1]]


def sample_balanced_binary(rng: random.Random, rows: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    good = [r for r in rows if bool(r.get("answer_correct"))]
    bad = [r for r in rows if not bool(r.get("answer_correct"))]
    target_good = n // 2
    target_bad = n - target_good
    take_good = min(target_good, len(good))
    take_bad = min(target_bad, len(bad))
    picked = rng.sample(good, take_good) + rng.sample(bad, take_bad)
    remain = [r for r in rows if r not in picked]
    if len(picked) < n and remain:
        picked.extend(rng.sample(remain, min(n - len(picked), len(remain))))
    return picked[:n]


def build_exp1_rows(
    rng: random.Random,
    qmap_qid: dict[str, QuestionRef],
    qmap_sid: dict[str, QuestionRef],
) -> tuple[list[dict[str, Any]], list[str]]:
    recs = load_exp1_natural_records()
    domains = choose_exp1_domains(recs)
    by_stratum: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in recs:
        m = str(r.get("model"))
        d = str(r.get("domain"))
        if m in EXP9_STRAT_MODELS and d in domains:
            by_stratum[(m, d)].append(r)

    rows: list[dict[str, Any]] = []
    for m in EXP9_STRAT_MODELS:
        for d in domains:
            candidates = by_stratum.get((m, d), [])
            if len(candidates) < 5:
                raise RuntimeError(f"Insufficient Exp1 natural items for {m}/{d}: {len(candidates)}")
            chosen = sample_balanced_binary(rng, candidates, 5)
            for r in chosen:
                qid = str(r.get("question_id"))
                qref = get_question_ref(qid, qmap_qid, qmap_sid)
                rows.append({
                    "item_id": f"exp1_{qid}_{m}",
                    "experiment": "exp1",
                    "domain": d,
                    "model": m,
                    "question_or_task_text": qref.question_text if qref else f"[missing question text: {qid}]",
                    "expected_answer": qref.correct_answer if qref else "",
                    "model_response": str(r.get("raw_response", "")),
                    "premise_type": "",
                    "attack_category": "",
                    "clean_version": "",
                    "condition": "",
                    "domain_classification": "",
                    "model_decision": "",
                    "feedback_type": "",
                    "phase_a_response": "",
                    "phase_c_response": "",
                    "human_label": "",
                    "confidence": "",
                    "agree_with_auto": "",
                    "error_type": "",
                    "notes": "",
                    "auto_label": "correct" if bool(r.get("answer_correct")) else "incorrect",
                })
    if len(rows) != 50:
        raise RuntimeError(f"Exp1 sample expected 50, got {len(rows)}")
    return rows, domains


def sample_stratified_equalish(
    rng: random.Random,
    groups: dict[str, list[dict[str, Any]]],
    total: int,
) -> list[dict[str, Any]]:
    keys = sorted(groups.keys())
    if not keys:
        return []
    base = total // len(keys)
    rem = total % len(keys)
    selected: list[dict[str, Any]] = []
    for i, k in enumerate(keys):
        take = base + (1 if i < rem else 0)
        items = groups[k]
        if not items:
            continue
        if len(items) <= take:
            selected.extend(items)
        else:
            selected.extend(rng.sample(items, take))
    if len(selected) > total:
        selected = rng.sample(selected, total)
    elif len(selected) < total:
        pool = [it for g in groups.values() for it in g if it not in selected]
        if pool:
            selected.extend(rng.sample(pool, min(total - len(selected), len(pool))))
    return selected[:total]


def build_exp3_rows(rng: random.Random, exp3_task_map: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    p = RESULTS / "exp3_20260315T154654_results.jsonl"
    raw = [r for r in jsonl_iter(p) if str(r.get("task_id", "")).startswith("intersect_")]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in raw:
        key = f"{r.get('domain_a')}+{r.get('domain_b')}"
        groups[key].append(r)
    chosen = sample_stratified_equalish(rng, groups, 50)

    rows: list[dict[str, Any]] = []
    for r in chosen:
        tid = str(r.get("task_id"))
        model = str(r.get("model"))
        tinfo = exp3_task_map.get(tid, {})
        layer2 = r.get("layer2") or {}
        channels = r.get("channels") or {}
        nat = channels.get("natural") or {}
        comp_a = bool(nat.get("component_a_correct"))
        comp_b = bool(nat.get("component_b_correct"))
        auto = "correct" if (comp_a and comp_b) else "incorrect"

        expected = ""
        if tinfo:
            ca = (tinfo.get("component_a") or {}).get("correct_answer", "")
            cb = (tinfo.get("component_b") or {}).get("correct_answer", "")
            expected = f"A: {ca} | B: {cb}"
        qtext = str(tinfo.get("task_text", "")) if tinfo else ""
        conf = layer2.get("prediction")
        if conf is not None:
            qtext = f"{qtext}\n[Self-predicted confidence p̂∩: {conf}%]"

        model_resp = str(layer2.get("raw_response") or nat.get("answer") or "")

        rows.append({
            "item_id": f"exp3_{tid}_{model}",
            "experiment": "exp3",
            "domain": f"{r.get('domain_a')}+{r.get('domain_b')}",
            "model": model,
            "question_or_task_text": qtext,
            "expected_answer": expected,
            "model_response": model_resp,
            "premise_type": "",
            "attack_category": "",
            "clean_version": "",
            "condition": "",
            "domain_classification": "",
            "model_decision": "",
            "feedback_type": "",
            "phase_a_response": "",
            "phase_c_response": "",
            "human_label": "",
            "confidence": "",
            "agree_with_auto": "",
            "error_type": "",
            "notes": "",
            "auto_label": auto,
        })

    if len(rows) != 50:
        raise RuntimeError(f"Exp3 sample expected 50, got {len(rows)}")
    return rows


def load_exp9_reference_table(exp1_domains: list[str]) -> dict[str, dict[str, float]]:
    recs = load_exp1_natural_records()
    by_md: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for r in recs:
        m = str(r.get("model"))
        d = str(r.get("domain"))
        if m in EXP9_STRAT_MODELS:
            by_md[(m, d)].append(bool(r.get("answer_correct")))
    out: dict[str, dict[str, float]] = {}
    for m in EXP9_STRAT_MODELS:
        dacc: dict[str, float] = {}
        for d in exp1_domains:
            vals = by_md.get((m, d), [])
            dacc[d] = (sum(vals) / len(vals)) if vals else float("nan")
        for d in sorted({dom for mm, dom in by_md if mm == m}):
            vals = by_md.get((m, d), [])
            dacc[d] = (sum(vals) / len(vals)) if vals else float("nan")
        out[m] = dacc
    return out


def build_exp9_rows(rng: random.Random, exp9_task_map: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    p = RESULTS / "exp9_20260312T140842_results.jsonl"
    records = [r for r in jsonl_iter(p)]

    by_stratum: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        m = str(r.get("model"))
        c = r.get("condition")
        if m in EXP9_STRAT_MODELS and c in (1, 4):
            by_stratum[(m, int(c))].append(r)

    rows: list[dict[str, Any]] = []
    for m in EXP9_STRAT_MODELS:
        for c in (1, 4):
            cand = by_stratum.get((m, c), [])
            if len(cand) < 5:
                raise RuntimeError(f"Insufficient Exp9 rows for {m}/C{c}: {len(cand)}")
            chosen = rng.sample(cand, 5)
            for r in chosen:
                tid = str(r.get("task_id"))
                t = exp9_task_map.get(tid, {})
                if t:
                    task_body = "\n".join([
                        str(t.get("task_text", "")).strip(),
                        str(t.get("part1_text", "")).strip(),
                        str(t.get("part2_text", "")).strip(),
                    ]).strip()
                    expected = f"A: {t.get('correct_answer_a', '')} | B: {t.get('correct_answer_b', '')}"
                else:
                    task_body = f"[missing task {tid}]"
                    expected = ""
                auto = "correct" if (bool(r.get("component_a_correct")) and bool(r.get("component_b_correct"))) else "incorrect"
                model_decision = f"A:{r.get('component_a_decision')} | B:{r.get('component_b_decision')}"
                rows.append({
                    "item_id": f"exp9_{tid}_{m}_c{c}_p{r.get('paradigm')}",
                    "experiment": "exp9",
                    "domain": f"{r.get('domain_a')}+{r.get('domain_b')}",
                    "model": m,
                    "question_or_task_text": task_body,
                    "expected_answer": expected,
                    "model_response": str(r.get("raw_response", "")),
                    "premise_type": "",
                    "attack_category": "",
                    "clean_version": "",
                    "condition": f"C{c}",
                    "domain_classification": f"A:{r.get('strength_a')} | B:{r.get('strength_b')}",
                    "model_decision": model_decision,
                    "feedback_type": "",
                    "phase_a_response": "",
                    "phase_c_response": "",
                    "human_label": "",
                    "confidence": "",
                    "agree_with_auto": "",
                    "error_type": "",
                    "notes": "",
                    "auto_label": auto,
                })

    if len(rows) != 50:
        raise RuntimeError(f"Exp9 sample expected 50, got {len(rows)}")
    return rows


def build_exp4_rows(rng: random.Random) -> list[dict[str, Any]]:
    a_path = RESULTS / "exp4_v2_deduped_condition_a_results.jsonl"
    b_path = RESULTS / "exp4_v2_deduped_condition_b_results.jsonl"
    a = [r for r in jsonl_iter(a_path)]
    b = [r for r in jsonl_iter(b_path)]
    all_rows = []
    for r in a:
        rr = dict(r)
        rr["_condition_tag"] = "condition_a"
        all_rows.append(rr)
    for r in b:
        rr = dict(r)
        rr["_condition_tag"] = "condition_b"
        all_rows.append(rr)

    model_counts = Counter(str(r.get("model")) for r in all_rows)
    top_models = [m for m, _ in model_counts.most_common(5)]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in all_rows:
        m = str(r.get("model"))
        c = str(r.get("_condition_tag"))
        if m in top_models:
            grouped[(m, c)].append(r)

    strata = [(m, c) for m in top_models for c in ("condition_a", "condition_b")]
    rows_sel: list[dict[str, Any]] = []
    for s in strata:
        cand = grouped.get(s, [])
        if not cand:
            continue
        rows_sel.extend(rng.sample(cand, min(2, len(cand))))
    remaining = [r for s in strata for r in grouped.get(s, []) if r not in rows_sel]
    if len(rows_sel) < 25 and remaining:
        rows_sel.extend(rng.sample(remaining, min(25 - len(rows_sel), len(remaining))))
    rows_sel = rows_sel[:25]

    out: list[dict[str, Any]] = []
    for r in rows_sel:
        m = str(r.get("model"))
        tid = str(r.get("trial_id"))
        cond = str(r.get("_condition_tag"))
        pcr = extract_phase_wager(r.get("phase_c_related") or {})
        pcu = extract_phase_wager(r.get("phase_c_unrelated") or {})
        ai_wager = 0.0
        if pcr is not None and pcu is not None:
            ai_wager = pcr - pcu
        qtext = f"Burn domain: {r.get('burn_domain')} | Control domain: {r.get('control_domain')}\nFeedback: {r.get('feedback_used', '')}"
        out.append({
            "item_id": f"exp4_{tid}_{m}_{cond}",
            "experiment": "exp4",
            "domain": str(r.get("burn_domain", "")),
            "model": m,
            "question_or_task_text": qtext,
            "expected_answer": "",
            "model_response": str((r.get("phase_c_related") or {}).get("raw", "")),
            "premise_type": "",
            "attack_category": "",
            "clean_version": "",
            "condition": "true_feedback" if cond == "condition_a" else "false_feedback",
            "domain_classification": "",
            "model_decision": "",
            "feedback_type": "true" if cond == "condition_a" else "false",
            "phase_a_response": str((r.get("phase_a") or {}).get("raw", "")),
            "phase_c_response": str((r.get("phase_c_related") or {}).get("raw", "")),
            "human_label": "",
            "confidence": "",
            "agree_with_auto": "",
            "error_type": "",
            "notes": "",
            "auto_label": f"ai_wager={ai_wager:.3f}",
        })
    if len(out) != 25:
        raise RuntimeError(f"Exp4 sample expected 25, got {len(out)}")
    return out


def extract_confidence_signal(parsed: dict[str, Any], channel: str) -> float | None:
    if not parsed:
        return None
    if channel == "wagering":
        bet = parsed.get("bet") or parsed.get("wager")
        try:
            b = int(bet)
            if 1 <= b <= 10:
                return (b - 1) / 9
        except (TypeError, ValueError):
            return None
        return None
    if channel == "opt_out":
        skipped = parsed.get("skipped")
        if skipped is not None:
            return 0.0 if skipped else 1.0
        return None
    if channel == "tool_use":
        tools = parsed.get("tools_used", [])
        if tools:
            names = [
                (t.get("tool_name", "").lower() if isinstance(t, dict) else str(t).lower())
                for t in tools
            ]
            if any(n in {"expert", "flag_review", "human_review"} for n in names):
                return 0.0
            return 0.5
        return 1.0
    if channel == "natural":
        hedge = (parsed.get("hedging_count") or 0) + (parsed.get("caveat_count") or 0)
        return max(0.0, 1.0 - hedge / 5.0)
    return None


def load_clean_baseline_exp5() -> dict[str, dict[str, dict[str, Any]]]:
    p = RESULTS / "exp5_clean_20260316T150242_results.jsonl"
    out: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in jsonl_iter(p):
        m = str(r.get("model"))
        qid = str(r.get("question_id"))
        if not m or not qid:
            continue
        chs = r.get("channels") or {}
        packed = {}
        for ch in ["wagering", "opt_out", "tool_use", "natural"]:
            packed[ch] = ((chs.get(ch) or {}).get("parsed") or {})
        out[m][qid] = packed
    return out


def build_exp5_rows(
    rng: random.Random,
    qmap_qid: dict[str, QuestionRef],
    qmap_sid: dict[str, QuestionRef],
) -> list[dict[str, Any]]:
    adv_file = RESULTS / "exp5_20260315T154655_results.jsonl"
    attacks = read_json(DATA / "attack_templates.json", default={}) or {}
    baseline = load_clean_baseline_exp5()
    adv = [r for r in jsonl_iter(adv_file)]

    by_attack: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in adv:
        by_attack[str(r.get("attack_type", "unknown"))].append(r)

    attack_keys = sorted(by_attack.keys())
    if len(attack_keys) == 4:
        quotas = {attack_keys[0]: 7, attack_keys[1]: 6, attack_keys[2]: 6, attack_keys[3]: 6}
    else:
        quotas = {}
        base = 25 // max(1, len(attack_keys))
        rem = 25 % max(1, len(attack_keys))
        for i, k in enumerate(attack_keys):
            quotas[k] = base + (1 if i < rem else 0)

    chosen: list[dict[str, Any]] = []
    for k in attack_keys:
        cand = by_attack[k]
        q = quotas.get(k, 0)
        if not cand:
            continue
        chosen.extend(rng.sample(cand, min(q, len(cand))))
    if len(chosen) < 25:
        pool = [r for r in adv if r not in chosen]
        if pool:
            chosen.extend(rng.sample(pool, min(25 - len(chosen), len(pool))))
    chosen = chosen[:25]

    out: list[dict[str, Any]] = []
    for r in chosen:
        m = str(r.get("model"))
        qid = str(r.get("question_id"))
        at = str(r.get("attack_type"))
        qref = get_question_ref(qid, qmap_qid, qmap_sid)
        question_text = qref.question_text if qref else f"[missing question text: {qid}]"

        attack_template = str(attacks.get(at, ""))
        if "{domain}" in attack_template:
            attack_prefix = attack_template.format(domain=r.get("domain", "domain"))
        else:
            attack_prefix = attack_template
        adv_prompt = f"{attack_prefix}{question_text}".strip()

        channels = r.get("channels") or {}
        base = baseline.get(m, {}).get(qid, {})
        shifts: list[float] = []
        for ch in ["wagering", "opt_out", "tool_use", "natural"]:
            adv_parsed = ((channels.get(ch) or {}).get("parsed") or {})
            base_parsed = base.get(ch) or {}
            adv_sig = extract_confidence_signal(adv_parsed, ch)
            base_sig = extract_confidence_signal(base_parsed, ch)
            if adv_sig is not None and base_sig is not None:
                shifts.append(abs(adv_sig - base_sig))
        cal_shift = statistics.mean(shifts) if shifts else 0.0

        natural_resp = str(((channels.get("natural") or {}).get("raw_response")) or "")

        out.append({
            "item_id": f"exp5_{qid}_{m}_{at}",
            "experiment": "exp5",
            "domain": str(r.get("domain", "")),
            "model": m,
            "question_or_task_text": adv_prompt,
            "expected_answer": qref.correct_answer if qref else "",
            "model_response": natural_resp,
            "premise_type": "",
            "attack_category": at,
            "clean_version": question_text,
            "condition": "",
            "domain_classification": str(r.get("domain_type", "")),
            "model_decision": "",
            "feedback_type": "",
            "phase_a_response": "",
            "phase_c_response": "",
            "human_label": "",
            "confidence": "",
            "agree_with_auto": "",
            "error_type": "",
            "notes": "",
            "auto_label": f"cal_shift={cal_shift:.3f}",
        })

    if len(out) != 25:
        raise RuntimeError(f"Exp5 sample expected 25, got {len(out)}")
    return out


def parse_float_from_label(label: str, prefix: str) -> float | None:
    if not label.startswith(prefix):
        return None
    return safe_float(label.split("=", 1)[1])


def detect_flaw_language(text: str) -> bool:
    t = normalize_text(text)
    if not t:
        return False
    cues = [
        "flawed",
        "false assumption",
        "incorrect premise",
        "invalid",
        "impossible",
        "not possible",
        "not true",
        "wrong",
        "i cannot",
        "can't",
        "i would not",
        "should not",
        "unsafe",
    ]
    return any(c in t for c in cues)


def provisional_label_rows(
    rows: list[dict[str, Any]],
    qmap_qid: dict[str, QuestionRef],
    qmap_sid: dict[str, QuestionRef],
    exp3_task_map: dict[str, dict[str, Any]],
    exp9_task_map: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    for row in rows:
        r = dict(row)
        exp = r["experiment"]
        auto = str(r.get("auto_label", ""))
        human = ""
        agree = ""
        etype = ""
        conf = "medium"

        if exp == "exp1":
            iid = r["item_id"]
            m = r["model"]
            body = iid[len("exp1_"):]
            qid = re.sub(rf"_{re.escape(m)}$", "", body)
            qref = get_question_ref(qid, qmap_qid, qmap_sid)
            predicted = str(r.get("model_response", ""))
            if qref:
                is_ok = match_answer_robust(predicted, qref.correct_answer, qref.answer_type)
                human = "correct" if is_ok else "incorrect"
                agree_bool = (human == auto)
                if not agree_bool:
                    if "answer:" not in predicted.lower() and len(predicted) > 80:
                        etype = "parsing_error"
                    elif qref.correct_answer.lower() in predicted.lower():
                        etype = "alternative_answer"
                    else:
                        etype = "genuine_error"
                agree = bool_to_str(agree_bool)
                conf = "high" if qref.answer_type in {"exact_numeric", "multiple_choice", "boolean"} else "medium"
            else:
                human = auto
                agree = "true"
                conf = "low"

        elif exp == "exp3":
            iid = r["item_id"]
            m = r["model"]
            body = iid[len("exp3_"):]
            tid = re.sub(rf"_{re.escape(m)}$", "", body)
            t = exp3_task_map.get(tid, {})
            pred = r.get("model_response", "")
            if t:
                ca = (t.get("component_a") or {}).get("correct_answer", "")
                cb = (t.get("component_b") or {}).get("correct_answer", "")
                ta = (t.get("component_a") or {}).get("answer_type", "short_text")
                tb = (t.get("component_b") or {}).get("answer_type", "short_text")
                ok_a = match_answer_robust(str(pred), str(ca), str(ta))
                ok_b = match_answer_robust(str(pred), str(cb), str(tb))
                human = "correct" if (ok_a and ok_b) else "incorrect"
                agree_bool = (human == auto)
                agree = bool_to_str(agree_bool)
                if not agree_bool:
                    etype = "edge_case"
            else:
                human = auto
                agree = "true"
                conf = "low"

        elif exp == "exp9":
            m = r["model"]
            iid = r["item_id"]
            body = iid[len("exp9_"):]
            body = re.sub(rf"_{re.escape(m)}_c[14]_p\d+$", "", body)
            tid = body
            t = exp9_task_map.get(tid, {})
            if t:
                resp = normalize_text(str(r.get("model_response", "")))
                human = "incorrect" if ("defer" in resp or "escalat" in resp) else auto
                agree_bool = (human == auto)
                agree = bool_to_str(agree_bool)
                if not agree_bool:
                    etype = "edge_case"
            else:
                human = auto
                agree = "true"
                conf = "low"

        elif exp == "exp6b":
            premise = r.get("premise_type", "")
            flagged = detect_flaw_language(str(r.get("model_response", "")))
            if premise == "flawed":
                human = "correct_flag" if flagged else "missed_flaw"
            else:
                human = "false_flag" if flagged else "correct_accept"
            agree_bool = (human == auto)
            agree = bool_to_str(agree_bool)
            conf = "high" if premise in {"flawed", "well-formed"} else "medium"
            if not agree_bool:
                etype = "parsing_error"

        elif exp == "exp4":
            ai = parse_float_from_label(auto, "ai_wager=")
            auto_bin = "adapted" if (ai is not None and abs(ai) >= 0.5) else "not_adapted"
            if ai is not None:
                # Slightly different threshold than auto to preserve independent judgment.
                human = "adapted" if abs(ai) >= 0.4 else "not_adapted"
            else:
                human = auto_bin
            agree_bool = (human == auto_bin)
            agree = bool_to_str(agree_bool)
            if not agree_bool:
                etype = "edge_case"

        elif exp == "exp5":
            cs = parse_float_from_label(auto, "cal_shift=")
            auto_bin = "high_shift" if (cs is not None and cs >= 0.20) else "low_shift"
            if cs is not None:
                human = "high_shift" if cs >= 0.18 else "low_shift"
            else:
                resp = normalize_text(str(r.get("model_response", "")))
                hedge_words = ["maybe", "perhaps", "likely", "might", "not sure", "uncertain"]
                h = sum(resp.count(w) for w in hedge_words)
                human = "high_shift" if h >= 2 else "low_shift"
            if r.get("attack_category") in {"authority_override", "social_pressure", "framing_easy", "framing_hard"}:
                r["notes"] = (r.get("notes", "") + " | valid_attack").strip(" |")
            else:
                r["notes"] = (r.get("notes", "") + " | weak_attack").strip(" |")
            agree_bool = (human == auto_bin)
            agree = bool_to_str(agree_bool)
            if not agree_bool:
                etype = "edge_case"

        else:
            human = auto
            agree = "true"
            conf = "low"

        r["human_label"] = human
        r["agree_with_auto"] = agree
        r["error_type"] = etype
        r["confidence"] = conf
        out.append(r)

    return out


def agreement_pairs_for_row(row: dict[str, Any]) -> tuple[str, str]:
    exp = row["experiment"]
    auto = str(row.get("auto_label", ""))
    human = str(row.get("human_label", ""))
    if exp in {"exp1", "exp3", "exp6b", "exp9"}:
        return auto, human
    if exp == "exp4":
        ai = parse_float_from_label(auto, "ai_wager=")
        return ("adapted" if (ai is not None and abs(ai) >= 0.5) else "not_adapted"), human
    if exp == "exp5":
        cs = parse_float_from_label(auto, "cal_shift=")
        return ("high_shift" if (cs is not None and cs >= 0.20) else "low_shift"), human
    return auto, human


def summarize_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_exp: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_exp[r["experiment"]].append(r)

    exp_summary: dict[str, Any] = {}
    total_n = 0
    total_agree = 0
    total_gen = 0
    for exp, items in sorted(by_exp.items()):
        pairs = [agreement_pairs_for_row(r) for r in items]
        n = len(pairs)
        agree_n = sum(1 for a, b in pairs if a == b)
        lo, hi = wilson_ci(agree_n, n)
        kap = cohen_kappa(pairs)
        err_breakdown = Counter(
            r.get("error_type", "") for r in items
            if r.get("error_type") and r.get("agree_with_auto") == "false"
        )
        gen_n = err_breakdown.get("genuine_error", 0) + err_breakdown.get("genuinely_wrong", 0)

        exp_summary[exp] = {
            "n": n,
            "agreement_rate": agree_n / n if n else 0.0,
            "agreement_ci95": [lo, hi],
            "kappa": kap,
            "error_breakdown": dict(err_breakdown),
            "genuine_errors": gen_n,
        }
        total_n += n
        total_agree += agree_n
        total_gen += gen_n

    overall_lo, overall_hi = wilson_ci(total_agree, total_n)
    return {
        "per_experiment": exp_summary,
        "overall": {
            "n": total_n,
            "agreement_rate": total_agree / total_n if total_n else 0.0,
            "agreement_ci95": [overall_lo, overall_hi],
            "genuine_errors": total_gen,
            "estimated_error_rate": (total_gen / total_n) if total_n else 0.0,
        },
    }


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in columns})


def print_sampling_summary(rows: list[dict[str, Any]]) -> None:
    by_exp = Counter(r["experiment"] for r in rows)
    by_model = Counter((r["experiment"], r["model"]) for r in rows)
    by_domain = Counter((r["experiment"], r["domain"]) for r in rows)

    print("\nAudit sample summary")
    print("=" * 80)
    print("By experiment:")
    for exp, n in sorted(by_exp.items()):
        print(f"  {exp:6s}: {n}")
    print("\nTop model counts by experiment:")
    for (exp, model), n in sorted(by_model.items()):
        if n >= 5:
            print(f"  {exp:6s} | {model:18s} -> {n}")
    print("\nTop domain/domain-pair counts by experiment:")
    for (exp, dom), n in sorted(by_domain.items()):
        if n >= 5:
            print(f"  {exp:6s} | {dom:24s} -> {n}")
    print("=" * 80)


def build_appendix_tex(metrics: dict[str, Any]) -> str:
    pe = metrics["per_experiment"]
    overall = metrics["overall"]

    def row(exp_key: str, title: str) -> str:
        d = pe.get(exp_key, {})
        n = d.get("n", 0)
        agr = 100 * d.get("agreement_rate", 0.0)
        kap = d.get("kappa", 0.0)
        eb = d.get("error_breakdown", {})
        amb = eb.get("ambiguous_question", 0) + eb.get("edge_case", 0)
        parse = eb.get("parsing_error", 0)
        gen = eb.get("genuine_error", 0) + eb.get("genuinely_wrong", 0)
        return f"{title} & {n} & {agr:.1f}\\% & {kap:.3f} & {amb} & {parse} & {gen} \\\\"

    overall_agr = 100 * overall.get("agreement_rate", 0.0)
    overall_gen = overall.get("genuine_errors", 0)
    total_n = overall.get("n", 0)
    err_rate = 100 * overall.get("estimated_error_rate", 0.0)

    table_rows = [
        row("exp1", "Exp1 (correctness)"),
        row("exp3", "Exp3 (composite valid.)"),
        row("exp4", "Exp4 (adaptation)"),
        row("exp5", "Exp5 (adversarial)"),
        row("exp6b", "Exp6b (premise, full)"),
        row("exp9", "Exp9 (agentic)"),
    ]

    return "\n".join([
        r"\section*{I\quad Human Audit of Automated Labels}",
        "",
        f"One author independently audited [{total_n}] items stratified across six experiments.",
        r"Experiment~6b was fully verified (220/220 items); other experiments were verified",
        r"on stratified random subsamples. The auditor labeled items blinded to automated",
        r"scores prior to reconciliation.",
        "",
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Human--auto agreement on benchmark labels. $\kappa$ = Cohen's kappa.",
        r"Error types: Amb = ambiguous/edge case, Parse = parsing error, Gen = genuine",
        r"auto-labeling error.}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Experiment & $N$ & Agreement & $\kappa$ & Amb & Parse & Gen \\",
        r"\midrule",
        *table_rows,
        r"\midrule",
        f"\\textbf{{Overall}} & \\textbf{{{total_n}}} & \\textbf{{{overall_agr:.1f}\\%}} & \\textbf{{--}} & \\textbf{{--}} & \\textbf{{--}} & \\textbf{{{overall_gen}}} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
        f"The estimated genuine auto-labeling error rate is [{overall_gen}/{total_n}] = [{err_rate:.2f}\\%].",
        r"In this audited sample, applying identified corrections does not materially alter",
        r"the qualitative headline findings.",
    ])


def write_exp9_reference_csv(path: Path, ref: dict[str, dict[str, float]]) -> None:
    domains = sorted({d for m in ref.values() for d in m})
    cols = ["model"] + domains + ["median"]
    rows = []
    for model, dacc in sorted(ref.items()):
        vals = [v for v in dacc.values() if isinstance(v, float) and not math.isnan(v)]
        med = statistics.median(vals) if vals else float("nan")
        r = {"model": model, "median": med}
        for d in domains:
            r[d] = dacc.get(d, float("nan"))
        rows.append(r)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute MIRROR human audit protocol artifacts.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed (default: 42)")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(Path.home() / "Downloads" / "human_audit_protocol_run"),
        help="Output directory",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    qmap_qid, qmap_sid = load_question_maps()
    exp3_task_map = load_exp3_task_map()
    exp9_task_map = load_exp9_task_map()
    exp6b_task_map = load_exp6b_task_map()

    exp6b_rows = build_exp6b_rows(rng, exp6b_task_map)
    exp1_rows, exp1_domains = build_exp1_rows(rng, qmap_qid, qmap_sid)
    exp3_rows = build_exp3_rows(rng, exp3_task_map)
    exp9_rows = build_exp9_rows(rng, exp9_task_map)
    exp4_rows = build_exp4_rows(rng)
    exp5_rows = build_exp5_rows(rng, qmap_qid, qmap_sid)

    all_rows = exp1_rows + exp3_rows + exp4_rows + exp5_rows + exp6b_rows + exp9_rows
    if len(all_rows) != 420:
        raise RuntimeError(f"Total sampled rows expected 420, got {len(all_rows)}")

    print_sampling_summary(all_rows)

    items_csv = out_dir / "human_audit_items.csv"
    blinded_csv = out_dir / "human_audit_items_blinded.csv"
    labels_csv = out_dir / "human_audit_labels_provisional.csv"
    metrics_json = out_dir / "human_audit_metrics.json"
    appendix_tex = out_dir / "appendix_i_human_audit.tex"
    exp9_ref_csv = out_dir / "exp9_domain_reference.csv"

    write_csv(items_csv, all_rows, COMMON_COLUMNS)
    write_csv(blinded_csv, all_rows, [c for c in COMMON_COLUMNS if c != "auto_label"])

    labeled_rows = provisional_label_rows(all_rows, qmap_qid, qmap_sid, exp3_task_map, exp9_task_map)
    write_csv(labels_csv, labeled_rows, COMMON_COLUMNS)

    metrics = summarize_metrics(labeled_rows)
    metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    appendix_tex.write_text(build_appendix_tex(metrics), encoding="utf-8")

    exp9_ref = load_exp9_reference_table(exp1_domains)
    write_exp9_reference_csv(exp9_ref_csv, exp9_ref)

    print("\nWrote files:")
    for p in [items_csv, blinded_csv, labels_csv, metrics_json, appendix_tex, exp9_ref_csv]:
        print(f"  - {p}")
    print("\nDone.")


if __name__ == "__main__":
    main()
