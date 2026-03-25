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

    def step_instance_baselines(self, state: dict) -> None:
        step = "instance_baselines"
        if step in state["completed_steps"]:
            return
        instance_run_id = f"{self.run_id}_instance"
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "exp9_instance_abstention_baselines.py"),
            "--run-id",
            instance_run_id,
            "--max-workers",
            str(max(2, self.max_workers)),
        ]
        proc = self.run_cmd(cmd, step=step)
        if proc.returncode != 0:
            raise RuntimeError(f"{step} failed:\n{proc.stderr}\n{proc.stdout}")
        instance_dir = ROOT / "data" / "results" / "exp9_instance_baselines" / instance_run_id
        state["instance_baseline_run_id"] = instance_run_id
        state["instance_baseline_summary_json"] = str(instance_dir / "instance_baseline_summary.json")
        state["instance_baseline_summary_md"] = str(instance_dir / "instance_baseline_summary.md")
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
        self.step_instance_baselines(state)
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
