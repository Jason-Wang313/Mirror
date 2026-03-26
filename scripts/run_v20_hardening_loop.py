"""
Run the MIRROR v20 human-data hardening pipeline with checkpoint/resume support.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PACKET_DIR = ROOT / "audit" / "human_baseline_packet"
RUNS_DIR = PACKET_DIR / "runs"
COHORT_DIR = PACKET_DIR / "cohort"
AUDIT_MULTI_DIR = PACKET_DIR / "audit_multirater"
DEPLOY_DIR = PACKET_DIR / "deployment"
REDTEAM_DIR = PACKET_DIR / "redteam"
DEPLOYMENT_PACKET_DIR = ROOT / "deployment_packet"
PAPER_DIR = ROOT / "paper"
PAPER_TEX = PAPER_DIR / "mirror_draft_v20.tex"
PAPER_PDF = PAPER_DIR / "mirror_draft_v20.pdf"
CALIBRATION_REPORT_PATH = Path(r"C:\Users\wangz\neurlips benchmark reviewing\data\mirror_v20_review_calibration_prompt_final.md")
HARD_V2_MANIFEST = PACKET_DIR / "hard_v2" / "packet_manifest.json"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def append_jsonl(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def row_count_csv(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def sorted_paths(pattern: str, base: Path) -> list[Path]:
    return sorted(base.glob(pattern), key=lambda p: p.name.lower())


def default_exp1_files() -> list[Path]:
    cohort = sorted_paths("exp1_response_sheet_H*.csv", COHORT_DIR)
    if cohort:
        return cohort
    return [
        PACKET_DIR / "templates" / "exp1_response_sheet_P1.csv",
        PACKET_DIR / "templates" / "exp1_response_sheet_P2.csv",
        PACKET_DIR / "templates" / "exp1_response_sheet_P3.csv",
    ]


def default_exp9_files() -> list[Path]:
    cohort = sorted_paths("exp9_response_sheet_H*.csv", COHORT_DIR)
    if cohort:
        return cohort
    return [
        PACKET_DIR / "templates" / "exp9_response_sheet_P1.csv",
        PACKET_DIR / "templates" / "exp9_response_sheet_P2.csv",
        PACKET_DIR / "templates" / "exp9_response_sheet_P3.csv",
    ]


def default_audit_files() -> list[Path]:
    raters = sorted_paths("real_human_audit_600_R*.csv", AUDIT_MULTI_DIR)
    if raters:
        return raters
    return [
        ROOT / "audit" / "real_human_audit_100_R1.csv",
        ROOT / "audit" / "real_human_audit_100_R2.csv",
        ROOT / "audit" / "real_human_audit_100_R3.csv",
    ]


def default_supporting_locked_files() -> list[Path]:
    files = [
        COHORT_DIR / "hard_packet_manifest.json",
        COHORT_DIR / "human_collection_manifest.json",
        DEPLOY_DIR / "ecological_validity_tasks.csv",
        DEPLOY_DIR / "ecological_validity_gold.csv",
        DEPLOY_DIR / "escalation_oracle_eval.csv",
        REDTEAM_DIR / "goodhart_attack_set.csv",
    ]
    return [p for p in files if p.exists()]


def participant_id_from_path(path: Path) -> str:
    stem = path.stem
    if "_H" in stem:
        return "H" + stem.rsplit("_H", 1)[1]
    if "_P" in stem:
        return "P" + stem.rsplit("_P", 1)[1]
    return stem


def default_participant_ids(exp1_files: list[Path]) -> list[str]:
    return [participant_id_from_path(p) for p in exp1_files]


def default_summary_stem(participant_ids: list[str]) -> str:
    if participant_ids and all(pid.startswith("H") for pid in participant_ids):
        return f"human_baseline_human{len(participant_ids)}_summary"
    return f"human_baseline_{len(participant_ids)}p_summary"


def default_cohort_label(participant_ids: list[str]) -> str:
    return f"Human Baseline ({len(participant_ids)} Participants)"


@dataclass
class Runner:
    run_id: str
    exp1_files: list[Path]
    exp9_files: list[Path]
    audit_files: list[Path]
    participant_ids: list[str]
    extra_locked_files: list[Path]
    summary_stem: str
    cohort_label: str
    max_workers: int
    verify_mapping_with_api: bool
    build_pdf: bool
    expected_exp1_rows: int | None
    expected_exp9_rows: int | None
    expected_audit_rows: int | None
    checkpoint_path: Path
    log_path: Path
    retry_queue_path: Path
    run_dir: Path
    results_dir: Path

    def load_state(self) -> dict:
        state = read_json(self.checkpoint_path)
        if not state:
            state = {
                "run_id": self.run_id,
                "created_at_utc": utc_now_iso(),
                "completed_steps": [],
                "completed_shards": [],
            }
        return state

    def save_state(self, state: dict) -> None:
        state["updated_at_utc"] = utc_now_iso()
        write_json(self.checkpoint_path, state)

    def run_cmd(self, cmd: list[str], step: str, cwd: Path = ROOT) -> subprocess.CompletedProcess:
        append_jsonl(
            self.log_path,
            {
                "ts_utc": utc_now_iso(),
                "event": "step_start",
                "step": step,
                "cmd": cmd,
            },
        )
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            text=True,
            capture_output=True,
        )
        append_jsonl(
            self.log_path,
            {
                "ts_utc": utc_now_iso(),
                "event": "step_end",
                "step": step,
                "returncode": proc.returncode,
                "stdout_tail": proc.stdout[-1500:],
                "stderr_tail": proc.stderr[-1500:],
            },
        )
        return proc

    def mark_step_complete(self, state: dict, step: str) -> None:
        if step not in state["completed_steps"]:
            state["completed_steps"].append(step)
            self.save_state(state)

    def mark_shard_complete(self, state: dict, shard: str) -> None:
        if shard not in state["completed_shards"]:
            state["completed_shards"].append(shard)
            self.save_state(state)

    def step_preflight_lock(self, state: dict) -> None:
        step = "preflight_lock"
        if step in state["completed_steps"]:
            return
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "human_input_preflight.py"),
            "--run-id",
            self.run_id,
            "--out-root",
            str(RUNS_DIR),
            "--exp1-files",
            *[str(p) for p in self.exp1_files],
            "--exp9-files",
            *[str(p) for p in self.exp9_files],
            "--audit-files",
            *[str(p) for p in self.audit_files],
        ]
        if self.extra_locked_files:
            cmd.extend(["--extra-locked-files", *[str(p) for p in self.extra_locked_files]])
        if self.expected_exp1_rows is not None:
            cmd.extend(["--exp1-rows", str(int(self.expected_exp1_rows))])
        if self.expected_exp9_rows is not None:
            cmd.extend(["--exp9-rows", str(int(self.expected_exp9_rows))])
        if self.expected_audit_rows is not None:
            cmd.extend(["--audit-rows", str(int(self.expected_audit_rows))])
        proc = self.run_cmd(cmd, step=step)
        if proc.returncode != 0:
            raise RuntimeError(f"{step} failed:\n{proc.stderr}\n{proc.stdout}")
        state["manifest_path"] = str(self.run_dir / "human_input_manifest.json")
        manifest = read_json(Path(state["manifest_path"]))
        policy = manifest.get("policy", {})
        state["audit_expected_rows"] = int(policy.get("audit_expected_rows", 100))
        self.mark_step_complete(state, step)

    def step_score_participant_shards(self, state: dict) -> None:
        step = "score_participant_shards"
        if step in state["completed_steps"]:
            return

        shards_dir = self.run_dir / "shards"
        shards_dir.mkdir(parents=True, exist_ok=True)
        failed_jobs = []

        def run_one_shard(pid: str, exp1_path: Path, exp9_path: Path) -> tuple[str, int, str]:
            shard_name = f"participant_{pid}"
            cmd = [
                sys.executable,
                str(ROOT / "scripts" / "score_human_baseline_packet.py"),
                "--exp1-responses",
                str(exp1_path),
                "--exp9-responses",
                str(exp9_path),
                "--participant-ids",
                pid,
                "--out-dir",
                str(shards_dir),
                "--summary-stem",
                shard_name,
                "--cohort-label",
                f"Human Baseline {pid}",
                "--weak-domain-rule",
                "median_or_bottom_k",
                "--fallback-bottom-k",
                "2",
            ]
            proc = self.run_cmd(cmd, step=f"{step}:{shard_name}")
            return shard_name, proc.returncode, proc.stderr[-1000:]

        pending = []
        for pid, exp1, exp9 in zip(self.participant_ids, self.exp1_files, self.exp9_files):
            shard_name = f"participant_{pid}"
            if shard_name in state["completed_shards"]:
                continue
            pending.append((pid, exp1, exp9))

        with ThreadPoolExecutor(max_workers=max(1, self.max_workers)) as ex:
            futures = {ex.submit(run_one_shard, pid, exp1, exp9): (pid, exp1, exp9) for pid, exp1, exp9 in pending}
            for fut in as_completed(futures):
                shard_name, code, stderr_tail = fut.result()
                if code == 0:
                    self.mark_shard_complete(state, shard_name)
                else:
                    failed_jobs.append(
                        {
                            "shard": shard_name,
                            "stderr_tail": stderr_tail,
                            "cmd_hint": f"re-run shard: {shard_name}",
                        }
                    )

        write_json(self.retry_queue_path, {"ts_utc": utc_now_iso(), "failed_jobs": failed_jobs})
        if failed_jobs:
            raise RuntimeError(f"{step} failed for {len(failed_jobs)} shard(s). See retry queue.")
        self.mark_step_complete(state, step)

    def step_score_aggregate(self, state: dict) -> None:
        step = "score_aggregate"
        if step in state["completed_steps"]:
            return
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "score_human_baseline_packet.py"),
            "--exp1-responses",
            *[str(p) for p in self.exp1_files],
            "--exp9-responses",
            *[str(p) for p in self.exp9_files],
            "--participant-ids",
            *self.participant_ids,
            "--out-dir",
            str(self.results_dir),
            "--summary-stem",
            self.summary_stem,
            "--cohort-label",
            self.cohort_label,
            "--weak-domain-rule",
            "median_or_bottom_k",
            "--fallback-bottom-k",
            "2",
            "--primary-aggregation",
            "participant_mean",
        ]
        proc = self.run_cmd(cmd, step=step)
        if proc.returncode != 0:
            raise RuntimeError(f"{step} failed:\n{proc.stderr}\n{proc.stdout}")
        self.mark_step_complete(state, step)

    def step_score_multirater_audit(self, state: dict) -> None:
        step = "score_multirater_audit"
        if step in state["completed_steps"]:
            return
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "score_human_audit_multirater.py"),
            "--rater-files",
            *[str(p) for p in self.audit_files],
            "--expected-rows",
            str(int(state.get("audit_expected_rows", row_count_csv(self.audit_files[0])))),
            "--out-dir",
            str(self.results_dir),
            "--summary-stem",
            "human_audit_multirater_summary",
        ]
        proc = self.run_cmd(cmd, step=step)
        if proc.returncode != 0:
            raise RuntimeError(f"{step} failed:\n{proc.stderr}\n{proc.stdout}")
        self.mark_step_complete(state, step)

    def step_verify_immutable(self, state: dict) -> None:
        step = "verify_immutable"
        if step in state["completed_steps"]:
            return
        manifest_path = Path(state["manifest_path"])
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "human_input_preflight.py"),
            "--verify-against-manifest",
            str(manifest_path),
        ]
        proc = self.run_cmd(cmd, step=step)
        if proc.returncode != 0:
            raise RuntimeError(f"{step} failed:\n{proc.stderr}\n{proc.stdout}")
        self.mark_step_complete(state, step)

    def step_context_hardening(self, state: dict) -> None:
        step = "context_hardening"
        if step in state["completed_steps"]:
            return
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "analyze_human_packet_context.py"),
            "--out-dir",
            str(self.results_dir),
        ]
        proc = self.run_cmd(cmd, step=step)
        if proc.returncode != 0:
            raise RuntimeError(f"{step} failed:\n{proc.stderr}\n{proc.stdout}")
        self.mark_step_complete(state, step)

    def step_hard_packet_v2_summary(self, state: dict) -> None:
        step = "hard_packet_v2_summary"
        if step in state["completed_steps"]:
            return
        if not HARD_V2_MANIFEST.exists():
            raise FileNotFoundError(f"Missing hard-v2 manifest: {HARD_V2_MANIFEST}")

        manifest = read_json(HARD_V2_MANIFEST)
        summary = {
            "run_id": self.run_id,
            "generated_at_utc": utc_now_iso(),
            "manifest_path": str(HARD_V2_MANIFEST),
            "exp1_items_total": manifest.get("exp1_items_total"),
            "exp9_items_total": manifest.get("exp9_items_total"),
            "exp1_per_domain": manifest.get("exp1_per_domain"),
            "exp9_per_pair": manifest.get("exp9_per_pair"),
            "exp1_empirical_hardness": manifest.get("exp1_empirical_hardness", {}),
            "exp9_bank_expansion": manifest.get("exp9_bank_expansion", {}),
            "status": "packet_prepared",
            "note": "Hard packet v2 is prepared and ready for participant collection/scoring.",
        }
        out_json = self.results_dir / "human_baseline_hardv2_summary.json"
        out_md = self.results_dir / "human_baseline_hardv2_summary.md"
        write_json(out_json, summary)
        md_lines = [
            "# Human Baseline Hard Packet v2 Summary",
            "",
            f"- Run ID: `{self.run_id}`",
            f"- Manifest: `{HARD_V2_MANIFEST}`",
            f"- Exp1 items total: {summary.get('exp1_items_total')}",
            f"- Exp9 items total: {summary.get('exp9_items_total')}",
            f"- Exp1 per-domain target: {summary.get('exp1_per_domain')}",
            f"- Exp9 per-pair target: {summary.get('exp9_per_pair')}",
            "",
            "## Packet Status",
            "",
            "- Hard packet v2 prepared with empirical-hardness Exp1 sampling and expanded Exp9 bank.",
            "- Reporting policy remains participant-mean primary, pooled sensitivity secondary after collection.",
            "",
        ]
        out_md.write_text("\n".join(md_lines), encoding="utf-8")
        state["hard_packet_v2_summary_json"] = str(out_json)
        state["hard_packet_v2_summary_md"] = str(out_md)
        self.save_state(state)
        self.mark_step_complete(state, step)

    def step_instance_baselines(self, state: dict) -> None:
        step = "instance_baselines"
        if step in state["completed_steps"]:
            return
        frames = [
            {
                "name": "legacy_c1_p3",
                "conditions": ["1"],
                "paradigms": ["3"],
                "label": "Condition 1 / Paradigm 3 (Legacy)",
            },
            {
                "name": "c1_all_paradigms",
                "conditions": ["1"],
                "paradigms": ["1", "2", "3"],
                "label": "Condition 1 / Paradigms 1-3",
            },
            {
                "name": "c1c2_all_paradigms",
                "conditions": ["1", "2"],
                "paradigms": ["1", "2", "3"],
                "label": "Condition 1-2 / Paradigms 1-3",
            },
        ]
        state["instance_baseline_runs"] = {}
        for frame in frames:
            frame_run_id = f"{self.run_id}_instance_{frame['name']}"
            cmd = [
                sys.executable,
                str(ROOT / "scripts" / "exp9_instance_abstention_baselines.py"),
                "--run-id",
                frame_run_id,
                "--max-workers",
                str(max(2, self.max_workers)),
                "--frame-label",
                frame["label"],
                "--conditions",
                *frame["conditions"],
                "--paradigms",
                *frame["paradigms"],
                "--strategy-set",
                "robust_v2",
                "--autonomy-grid",
                "0.30",
                "0.40",
                "0.50",
                "0.60",
                "0.70",
                "0.80",
                "0.90",
                "--conformal-target-grid",
                "0.10",
                "0.15",
                "0.20",
                "0.25",
                "0.30",
                "--calibration-method",
                "transformed_platt",
                "--matched-budget-mode",
                "domain_budget",
            ]
            proc = self.run_cmd(cmd, step=f"{step}:{frame['name']}")
            if proc.returncode != 0:
                raise RuntimeError(f"{step} failed for frame {frame['name']}:\n{proc.stderr}\n{proc.stdout}")
            instance_dir = ROOT / "data" / "results" / "exp9_instance_baselines" / frame_run_id
            state["instance_baseline_runs"][frame["name"]] = {
                "run_id": frame_run_id,
                "summary_json": str(instance_dir / "instance_baseline_summary.json"),
                "summary_md": str(instance_dir / "instance_baseline_summary.md"),
            }

        # Backward-compatible aliases point to the expanded C1 all-paradigm frame.
        primary = state["instance_baseline_runs"]["c1_all_paradigms"]
        state["instance_baseline_run_id"] = primary["run_id"]
        state["instance_baseline_summary_json"] = primary["summary_json"]
        state["instance_baseline_summary_md"] = primary["summary_md"]
        self.save_state(state)
        self.mark_step_complete(state, step)

    def step_proper_scoring_primary(self, state: dict) -> None:
        step = "proper_scoring_primary"
        if step in state["completed_steps"]:
            return
        proper_run_id = f"{self.run_id}_proper"
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "exp1_proper_scoring_primary.py"),
            "--run-id",
            proper_run_id,
        ]
        proc = self.run_cmd(cmd, step=step)
        if proc.returncode != 0:
            raise RuntimeError(f"{step} failed:\n{proc.stderr}\n{proc.stdout}")
        proper_dir = ROOT / "data" / "results" / "exp1_proper_scoring" / proper_run_id
        state["proper_scoring_run_id"] = proper_run_id
        state["proper_scoring_summary_json"] = str(proper_dir / "proper_scoring_primary_summary.json")
        state["proper_scoring_summary_md"] = str(proper_dir / "proper_scoring_primary_summary.md")
        self.save_state(state)
        self.mark_step_complete(state, step)

    def step_parse_missingness(self, state: dict) -> None:
        step = "parse_missingness"
        if step in state["completed_steps"]:
            return
        parse_run_id = f"{self.run_id}_parse_missingness"
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "analyze_parse_missingness.py"),
            "--run-id",
            parse_run_id,
            "--out-root",
            str(ROOT / "audit" / "human_baseline_packet" / "runs"),
            "--imputation",
            "all",
            "--mnar-bound-mode",
            "moderate",
            "--exp3-cce-path",
            str(ROOT / "audit" / "human_baseline_packet" / "results" / "exp3_difficulty_control_summary.json"),
        ]
        proc = self.run_cmd(cmd, step=step)
        if proc.returncode != 0:
            raise RuntimeError(f"{step} failed:\n{proc.stderr}\n{proc.stdout}")
        parse_dir = ROOT / "audit" / "human_baseline_packet" / "runs" / parse_run_id
        state["parse_missingness_run_id"] = parse_run_id
        state["parse_missingness_summary_json"] = str(parse_dir / "parse_missingness_summary.json")
        state["parse_missingness_summary_md"] = str(parse_dir / "parse_missingness_summary.md")
        self.save_state(state)
        self.mark_step_complete(state, step)

    def step_exp3_difficulty_control(self, state: dict) -> None:
        step = "exp3_difficulty_control"
        if step in state["completed_steps"]:
            return
        run_id = f"{self.run_id}_exp3_difficulty"
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "analyze_exp3_difficulty_control.py"),
            "--run-id",
            run_id,
            "--out-root",
            str(ROOT / "audit" / "human_baseline_packet" / "runs"),
            "--bootstrap-samples",
            "1200",
        ]
        proc = self.run_cmd(cmd, step=step)
        if proc.returncode != 0:
            raise RuntimeError(f"{step} failed:\n{proc.stderr}\n{proc.stdout}")
        out_dir = ROOT / "audit" / "human_baseline_packet" / "runs" / run_id
        state["exp3_difficulty_control_run_id"] = run_id
        state["exp3_difficulty_control_summary_json"] = str(out_dir / "exp3_difficulty_control_summary.json")
        state["exp3_difficulty_control_summary_md"] = str(out_dir / "exp3_difficulty_control_summary.md")
        self.save_state(state)
        self.mark_step_complete(state, step)

    def step_mci_channel_robustness(self, state: dict) -> None:
        step = "mci_channel_robustness"
        if step in state["completed_steps"]:
            return
        run_id = f"{self.run_id}_mci_robustness"
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "analyze_mci_channel_robustness.py"),
            "--run-id",
            run_id,
            "--out-root",
            str(ROOT / "audit" / "human_baseline_packet" / "runs"),
            "--min-shared-for-corr",
            "30",
            "--leave-one-out",
        ]
        proc = self.run_cmd(cmd, step=step)
        if proc.returncode != 0:
            raise RuntimeError(f"{step} failed:\n{proc.stderr}\n{proc.stdout}")
        out_dir = ROOT / "audit" / "human_baseline_packet" / "runs" / run_id
        state["mci_channel_robustness_run_id"] = run_id
        state["mci_channel_robustness_summary_json"] = str(out_dir / "mci_channel_robustness_summary.json")
        state["mci_channel_robustness_summary_md"] = str(out_dir / "mci_channel_robustness_summary.md")
        self.save_state(state)
        self.mark_step_complete(state, step)

    def step_non_oracle_utility(self, state: dict) -> None:
        step = "non_oracle_utility"
        if step in state["completed_steps"]:
            return
        run_id = f"{self.run_id}_non_oracle"
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "analyze_non_oracle_utility.py"),
            "--run-id",
            run_id,
            "--out-root",
            str(ROOT / "audit" / "human_baseline_packet" / "runs"),
            "--emit-pareto",
            "--cost-column",
            "cost_usd",
            "--latency-column",
            "total_latency_ms",
            "--resolver-q-grid",
            "0.50",
            "0.60",
            "0.70",
            "0.80",
            "0.90",
            "1.00",
        ]
        proc = self.run_cmd(cmd, step=step)
        if proc.returncode != 0:
            raise RuntimeError(f"{step} failed:\n{proc.stderr}\n{proc.stdout}")
        out_dir = ROOT / "audit" / "human_baseline_packet" / "runs" / run_id
        state["non_oracle_utility_run_id"] = run_id
        state["non_oracle_utility_summary_json"] = str(out_dir / "non_oracle_utility_summary.json")
        state["non_oracle_utility_summary_md"] = str(out_dir / "non_oracle_utility_summary.md")
        self.save_state(state)
        self.mark_step_complete(state, step)

    def step_weak_domain_policy_frontier(self, state: dict) -> None:
        step = "weak_domain_policy_frontier"
        if step in state["completed_steps"]:
            return
        run_id = f"{self.run_id}_policy_frontier"
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "analyze_exp9_weak_domain_policy_frontier.py"),
            "--run-id",
            run_id,
            "--out-root",
            str(ROOT / "audit" / "human_baseline_packet" / "runs"),
            "--conditions",
            "1",
            "--paradigms",
            "1",
            "2",
            "3",
            "--fallback-bottom-k",
            "2",
            "--bottom-k-grid",
            "1",
            "2",
            "3",
            "4",
            "--absolute-grid",
            "0.35",
            "0.40",
            "0.45",
            "0.50",
            "0.55",
            "0.60",
            "--quantile-grid",
            "0.20",
            "0.25",
            "0.30",
            "0.40",
            "0.50",
            "--deployment-packet-dir",
            str(DEPLOYMENT_PACKET_DIR),
        ]
        proc = self.run_cmd(cmd, step=step)
        if proc.returncode != 0:
            raise RuntimeError(f"{step} failed:\n{proc.stderr}\n{proc.stdout}")
        out_dir = ROOT / "audit" / "human_baseline_packet" / "runs" / run_id
        state["policy_frontier_run_id"] = run_id
        state["policy_frontier_summary_json"] = str(out_dir / "exp9_policy_frontier_summary.json")
        state["policy_frontier_summary_md"] = str(out_dir / "exp9_policy_frontier_summary.md")
        frontier_json = Path(state["policy_frontier_summary_json"])
        frontier_md = Path(state["policy_frontier_summary_md"])
        if frontier_json.exists():
            shutil.copy2(frontier_json, self.results_dir / "exp9_policy_frontier_summary.json")
        if frontier_md.exists():
            shutil.copy2(frontier_md, self.results_dir / "exp9_policy_frontier_summary.md")
        self.save_state(state)
        self.mark_step_complete(state, step)

    def step_build_pdf(self, state: dict) -> None:
        step = "build_pdf"
        if step in state["completed_steps"]:
            return
        if not self.build_pdf:
            self.mark_step_complete(state, step)
            return

        pdflatex_exe = shutil.which("pdflatex")
        if not pdflatex_exe:
            candidate = Path.home() / ".local" / "bin" / "pdflatex.cmd"
            if candidate.exists():
                pdflatex_exe = str(candidate)
        if not pdflatex_exe:
            raise FileNotFoundError("pdflatex executable not found in PATH or ~/.local/bin/pdflatex.cmd")

        bibtex_exe = shutil.which("bibtex")
        if not bibtex_exe:
            candidate_bib = Path.home() / ".local" / "bin" / "bibtex.cmd"
            if candidate_bib.exists():
                bibtex_exe = str(candidate_bib)

        pdflatex_cmd = [pdflatex_exe, "-interaction=nonstopmode", "-halt-on-error", PAPER_TEX.name]
        proc1 = self.run_cmd(pdflatex_cmd, step=f"{step}:pass1", cwd=PAPER_DIR)
        if proc1.returncode != 0:
            raise RuntimeError(f"{step} pass1 failed:\n{proc1.stderr}\n{proc1.stdout}")

        aux = PAPER_DIR / "mirror_draft_v20.aux"
        if aux.exists() and "\\citation" in aux.read_text(encoding="utf-8", errors="ignore"):
            if bibtex_exe:
                bib_proc = self.run_cmd([bibtex_exe, "mirror_draft_v20"], step=f"{step}:bibtex", cwd=PAPER_DIR)
                if bib_proc.returncode != 0:
                    raise RuntimeError(f"{step} bibtex failed:\n{bib_proc.stderr}\n{bib_proc.stdout}")
            else:
                append_jsonl(
                    self.log_path,
                    {
                        "ts_utc": utc_now_iso(),
                        "event": "step_warning",
                        "step": f"{step}:bibtex",
                        "warning": "bibtex executable not found; continuing with pdflatex-only passes",
                    },
                )

        proc2 = self.run_cmd(pdflatex_cmd, step=f"{step}:pass2", cwd=PAPER_DIR)
        if proc2.returncode != 0:
            raise RuntimeError(f"{step} pass2 failed:\n{proc2.stderr}\n{proc2.stdout}")
        proc3 = self.run_cmd(pdflatex_cmd, step=f"{step}:pass3", cwd=PAPER_DIR)
        if proc3.returncode != 0:
            raise RuntimeError(f"{step} pass3 failed:\n{proc3.stderr}\n{proc3.stdout}")

        if not PAPER_PDF.exists():
            raise FileNotFoundError(f"Expected PDF missing after build: {PAPER_PDF}")
        state["paper_pdf_path"] = str(PAPER_PDF)
        self.save_state(state)
        self.mark_step_complete(state, step)

    def step_exp9_mapping_validity(self, state: dict) -> None:
        step = "exp9_mapping_validity"
        if step in state["completed_steps"]:
            return
        run_id = f"{self.run_id}_mapping"
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "analyze_exp9_mapping_validity.py"),
            "--run-id",
            run_id,
            "--out-root",
            str(ROOT / "audit" / "human_baseline_packet" / "runs"),
            "--checkpoint-dir",
            str(ROOT / "data" / "exp9_verification_runs"),
            "--max-workers",
            str(max(2, self.max_workers)),
            "--fixed-only",
            "--resume",
        ]
        if self.verify_mapping_with_api:
            cmd.append("--verify-with-api")
        proc = self.run_cmd(cmd, step=step)
        if proc.returncode != 0:
            raise RuntimeError(f"{step} failed:\n{proc.stderr}\n{proc.stdout}")
        out_dir = ROOT / "audit" / "human_baseline_packet" / "runs" / run_id
        state["exp9_mapping_validity_run_id"] = run_id
        state["exp9_mapping_validity_summary_json"] = str(out_dir / "exp9_mapping_validity_summary.json")
        state["exp9_mapping_validity_summary_md"] = str(out_dir / "exp9_mapping_validity_summary.md")
        self.save_state(state)
        self.mark_step_complete(state, step)

    def step_calibration_gate(self, state: dict) -> None:
        step = "calibration_gate"
        if step in state["completed_steps"]:
            return
        out = {
            "source_path": str(CALIBRATION_REPORT_PATH),
            "found": CALIBRATION_REPORT_PATH.exists(),
            "generated_at_utc": utc_now_iso(),
            "actionable_items": [],
            "status": "missing_source",
        }
        if CALIBRATION_REPORT_PATH.exists():
            text = CALIBRATION_REPORT_PATH.read_text(encoding="utf-8", errors="ignore")
            action_section = ""
            if "## Step 5: Actionable revision list" in text:
                action_section = text.split("## Step 5: Actionable revision list", 1)[1]
                action_section = action_section.split("## Constraints compliance check", 1)[0]
            lines = [ln.strip() for ln in action_section.splitlines() if ln.strip()]
            items = [ln for ln in lines if ln.startswith("### ")]
            blocked = [
                ln for ln in items
                if any(tok in ln.lower() for tok in ["blocked", "scope-blocked", "execution-blocked", "requires api", "requires real log"])
            ]
            fixable_now = [ln for ln in items if ln not in blocked]
            out["actionable_items"] = items
            out["blocked_items"] = blocked
            out["fixable_now_items"] = fixable_now
            out["status"] = "ok"
            out["n_actionable_items"] = len(items)
            out["n_blocked_items"] = len(blocked)
            out["n_fixable_now_items"] = len(fixable_now)
        gate_json = self.results_dir / "calibration_gate_summary.json"
        gate_md = self.results_dir / "calibration_gate_summary.md"
        write_json(gate_json, out)
        md_lines = [
            "# Calibration Gate Summary",
            "",
            f"- Source found: {out['found']}",
            f"- Source path: `{out['source_path']}`",
            f"- Status: {out['status']}",
            f"- Actionable items: {out.get('n_actionable_items', 0)}",
            f"- Fixable-now items: {out.get('n_fixable_now_items', 0)}",
            f"- Blocked items: {out.get('n_blocked_items', 0)}",
            "",
        ]
        if out.get("actionable_items"):
            md_lines.append("## Parsed Items")
            md_lines.append("")
            for item in out["actionable_items"]:
                md_lines.append(f"- {item}")
        if out.get("fixable_now_items"):
            md_lines.extend(["", "## Fixable-Now", ""])
            for item in out["fixable_now_items"]:
                md_lines.append(f"- {item}")
        if out.get("blocked_items"):
            md_lines.extend(["", "## Blocked", ""])
            for item in out["blocked_items"]:
                md_lines.append(f"- {item}")
        gate_md.write_text("\n".join(md_lines), encoding="utf-8")
        state["calibration_gate_summary_json"] = str(gate_json)
        state["calibration_gate_summary_md"] = str(gate_md)
        self.save_state(state)
        self.mark_step_complete(state, step)

    def step_stanford_checklist_gate(self, state: dict) -> None:
        step = "stanford_checklist_gate"
        if step in state["completed_steps"]:
            return
        run_id = f"{self.run_id}_stanford_gate"
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "stanford_feedback_checklist.py"),
            "--run-id",
            run_id,
            "--out-root",
            str(ROOT / "audit" / "human_baseline_packet" / "runs"),
            "--results-dir",
            str(self.results_dir),
        ]
        proc = self.run_cmd(cmd, step=step)
        if proc.returncode != 0:
            raise RuntimeError(f"{step} failed:\n{proc.stderr}\n{proc.stdout}")
        out_dir = ROOT / "audit" / "human_baseline_packet" / "runs" / run_id
        state["stanford_checklist_run_id"] = run_id
        state["stanford_checklist_json"] = str(out_dir / "stanford_feedback_checklist.json")
        state["stanford_checklist_md"] = str(out_dir / "stanford_feedback_checklist.md")
        self.save_state(state)
        self.mark_step_complete(state, step)

    def step_exp3_mci_stability(self, state: dict) -> None:
        step = "exp3_mci_stability"
        if step in state["completed_steps"]:
            return
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "analyze_exp3_mci_stability.py"),
            "--out-dir",
            str(self.results_dir),
        ]
        proc = self.run_cmd(cmd, step=step)
        if proc.returncode != 0:
            raise RuntimeError(f"{step} failed:\n{proc.stderr}\n{proc.stdout}")
        state["exp3_mci_stability_json"] = str(self.results_dir / "exp3_mci_stability_summary.json")
        state["exp3_mci_stability_md"] = str(self.results_dir / "exp3_mci_stability_summary.md")
        self.save_state(state)
        self.mark_step_complete(state, step)

    def step_deployment_packet_validation(self, state: dict) -> None:
        step = "deployment_packet_validation"
        if step in state["completed_steps"]:
            return
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "validate_deployment_packet.py"),
            "--packet-dir",
            str(DEPLOYMENT_PACKET_DIR),
            "--out-dir",
            str(self.results_dir),
            "--allow-missing",
        ]
        proc = self.run_cmd(cmd, step=step)
        if proc.returncode != 0:
            raise RuntimeError(f"{step} failed:\n{proc.stderr}\n{proc.stdout}")
        state["deployment_packet_summary_json"] = str(self.results_dir / "deployment_packet_summary.json")
        state["deployment_packet_validation_md"] = str(self.results_dir / "validation_report.md")
        self.save_state(state)
        self.mark_step_complete(state, step)

    def step_promote_outputs(self, state: dict) -> None:
        step = "promote_outputs"
        if step in state["completed_steps"]:
            return
        stable_results = PACKET_DIR / "results"
        stable_results.mkdir(parents=True, exist_ok=True)

        promote_files = [
            f"{self.summary_stem}.json",
            f"{self.summary_stem}.md",
            "human_audit_multirater_summary.json",
            "human_audit_multirater_summary.md",
            "ecological_validity_summary.json",
            "oracle_realism_sensitivity.json",
            "goodhart_redteam_summary.json",
            "context_hardening_summary.md",
            "exp3_mci_stability_summary.json",
            "exp3_mci_stability_summary.md",
            "deployment_packet_summary.json",
            "validation_report.md",
            "source_manifest.json",
            "data_profile.json",
            "calibration_gate_summary.json",
            "calibration_gate_summary.md",
        ]
        for name in promote_files:
            src = self.results_dir / name
            dst = stable_results / name
            if not src.exists():
                raise FileNotFoundError(f"Expected output missing: {src}")
            shutil.copy2(src, dst)

        instance_json = Path(state.get("instance_baseline_summary_json", ""))
        instance_md = Path(state.get("instance_baseline_summary_md", ""))
        if instance_json.exists():
            shutil.copy2(instance_json, stable_results / "exp9_instance_baseline_summary.json")
        if instance_md.exists():
            shutil.copy2(instance_md, stable_results / "exp9_instance_baseline_summary.md")

        # Promote all instance-baseline frames.
        for frame_name, meta in state.get("instance_baseline_runs", {}).items():
            j = Path(meta.get("summary_json", ""))
            m = Path(meta.get("summary_md", ""))
            if j.exists():
                shutil.copy2(j, stable_results / f"exp9_instance_baseline_summary_{frame_name}.json")
            if m.exists():
                shutil.copy2(m, stable_results / f"exp9_instance_baseline_summary_{frame_name}.md")

        proper_json = Path(state.get("proper_scoring_summary_json", ""))
        proper_md = Path(state.get("proper_scoring_summary_md", ""))
        if proper_json.exists():
            shutil.copy2(proper_json, stable_results / "exp1_proper_scoring_primary_summary.json")
        if proper_md.exists():
            shutil.copy2(proper_md, stable_results / "exp1_proper_scoring_primary_summary.md")

        extra_promotions = [
            ("parse_missingness_summary_json", "parse_missingness_summary.json"),
            ("parse_missingness_summary_md", "parse_missingness_summary.md"),
            ("exp3_difficulty_control_summary_json", "exp3_difficulty_control_summary.json"),
            ("exp3_difficulty_control_summary_md", "exp3_difficulty_control_summary.md"),
            ("mci_channel_robustness_summary_json", "mci_channel_robustness_summary.json"),
            ("mci_channel_robustness_summary_md", "mci_channel_robustness_summary.md"),
            ("non_oracle_utility_summary_json", "non_oracle_utility_summary.json"),
            ("non_oracle_utility_summary_md", "non_oracle_utility_summary.md"),
            ("policy_frontier_summary_json", "exp9_policy_frontier_summary.json"),
            ("policy_frontier_summary_md", "exp9_policy_frontier_summary.md"),
            ("exp9_mapping_validity_summary_json", "exp9_mapping_validity_summary.json"),
            ("exp9_mapping_validity_summary_md", "exp9_mapping_validity_summary.md"),
            ("stanford_checklist_json", "stanford_feedback_checklist.json"),
            ("stanford_checklist_md", "stanford_feedback_checklist.md"),
        ]
        for state_key, dst_name in extra_promotions:
            src = Path(state.get(state_key, ""))
            if src.exists():
                shutil.copy2(src, stable_results / dst_name)

        # Stable aliases for downstream manuscript tooling.
        shutil.copy2(
            self.results_dir / f"{self.summary_stem}.json",
            stable_results / "human_baseline_summary_latest.json",
        )
        shutil.copy2(
            self.results_dir / f"{self.summary_stem}.md",
            stable_results / "human_baseline_summary_latest.md",
        )

        self.mark_step_complete(state, step)

    def run(self) -> None:
        state = self.load_state()
        self.save_state(state)

        self.step_preflight_lock(state)
        self.step_score_participant_shards(state)
        self.step_score_aggregate(state)
        self.step_score_multirater_audit(state)
        self.step_verify_immutable(state)
        self.step_context_hardening(state)
        self.step_hard_packet_v2_summary(state)
        self.step_instance_baselines(state)
        self.step_proper_scoring_primary(state)
        self.step_parse_missingness(state)
        self.step_exp3_difficulty_control(state)
        self.step_mci_channel_robustness(state)
        self.step_non_oracle_utility(state)
        self.step_weak_domain_policy_frontier(state)
        self.step_exp9_mapping_validity(state)
        self.step_exp3_mci_stability(state)
        self.step_deployment_packet_validation(state)
        self.step_build_pdf(state)
        self.step_calibration_gate(state)
        self.step_stanford_checklist_gate(state)
        self.step_promote_outputs(state)

        append_jsonl(
            self.log_path,
            {
                "ts_utc": utc_now_iso(),
                "event": "run_complete",
                "run_id": self.run_id,
                "checkpoint": str(self.checkpoint_path),
            },
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MIRROR v20 human-data hardening pipeline.")
    parser.add_argument("--run-id", type=str, default=datetime.now().strftime("v20_hardening_%Y%m%dT%H%M%S"))
    parser.add_argument("--exp1-files", nargs="+", type=Path, default=default_exp1_files())
    parser.add_argument("--exp9-files", nargs="+", type=Path, default=default_exp9_files())
    parser.add_argument("--audit-files", nargs="+", type=Path, default=default_audit_files())
    parser.add_argument("--participant-ids", nargs="+", type=str, default=None)
    parser.add_argument("--summary-stem", type=str, default=None)
    parser.add_argument("--cohort-label", type=str, default=None)
    parser.add_argument("--extra-locked-files", nargs="+", type=Path, default=default_supporting_locked_files())
    parser.add_argument("--max-workers", type=int, default=min(8, max(2, (os.cpu_count() or 4) // 2)))
    parser.add_argument("--verify-mapping-with-api", action="store_true", default=False)
    parser.add_argument("--skip-pdf-build", action="store_true", default=False)
    parser.add_argument("--exp1-rows", type=int, default=None)
    parser.add_argument("--exp9-rows", type=int, default=None)
    parser.add_argument("--audit-rows", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    participant_ids = args.participant_ids if args.participant_ids is not None else default_participant_ids(args.exp1_files)
    if len(args.exp1_files) != len(args.exp9_files):
        raise ValueError("exp1 and exp9 file lists must have equal length.")
    if len(participant_ids) != len(args.exp1_files):
        raise ValueError("participant_ids length must match response files.")
    summary_stem = args.summary_stem or default_summary_stem(participant_ids)
    cohort_label = args.cohort_label or default_cohort_label(participant_ids)

    run_dir = RUNS_DIR / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    runner = Runner(
        run_id=args.run_id,
        exp1_files=args.exp1_files,
        exp9_files=args.exp9_files,
        audit_files=args.audit_files,
        participant_ids=participant_ids,
        extra_locked_files=args.extra_locked_files,
        summary_stem=summary_stem,
        cohort_label=cohort_label,
        max_workers=max(1, args.max_workers),
        verify_mapping_with_api=bool(args.verify_mapping_with_api),
        build_pdf=not bool(args.skip_pdf_build),
        expected_exp1_rows=args.exp1_rows,
        expected_exp9_rows=args.exp9_rows,
        expected_audit_rows=args.audit_rows,
        checkpoint_path=run_dir / "checkpoint.json",
        log_path=run_dir / "progress_log.jsonl",
        retry_queue_path=run_dir / "retry_queue.json",
        run_dir=run_dir,
        results_dir=results_dir,
    )
    runner.run()
    print(f"Run complete: {run_dir}")


if __name__ == "__main__":
    main()
