#!/usr/bin/env python3
"""
Comprehensive test suite for lm_eval_harness.sh container migration.

Tests all backends, GPU configurations, eval types, path mappings, and edge cases.

Usage:
    python test_container_migration.py --dry-run          # Show what would run
    python test_container_migration.py --run              # Submit jobs
    python test_container_migration.py --run --tier core  # Run only core tests
    python test_container_migration.py --check            # Check job results
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

# =============================================================================
# CONFIGURATION - UPDATE THESE
# =============================================================================

# Models
MODEL_SMALL = "meta-llama/Llama-3.2-1B"          # Small, public, fast
MODEL_LARGE = "meta-llama/Llama-3.1-8B-Instruct" # Larger, gated, needs HF_TOKEN
MODEL_LOCAL = "/scratch/project_462000963/cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9"  # Local cached model

# Output
OUTPUT_ROOT = "./test_output/container_migration"

# SLURM
PARTITION = "dev-g"  # 2-job limit but faster queue
TIME_SHORT = "00:30:00"
TIME_LONG = "01:00:00"

# Test size
LIMIT = 20  # Samples per task

# =============================================================================
# COMPREHENSIVE TEST STRATEGY - 3 TIERS
# =============================================================================
#
# TIER 1 - CORE (Must pass for merge): 5 jobs
#   - vLLM backend basics (1 GPU)
#   - HF backend parity (1 GPU)
#   - Multi-GPU tensor parallelism (8 GPU)
#   - Dummy backend (no GPU)
#   - Custom config (lm_eval repo, transformers version)
#
# TIER 2 - EXTENDED (Recommended): 6 jobs
#   - Multilingual evals (Finnish, Swedish)
#   - Generation tasks (lambada, triviaqa)
#   - 4 GPU intermediate config
#   - Chat template variations
#   - Multiple evals in single job
#   - Local model path (if configured)
#
# TIER 3 - EDGE CASES (Optional): 4 jobs
#   - Trust remote code
#   - Custom tokenizer
#   - Various model_args combinations
#   - Batch size variations
#
# =============================================================================

@dataclass
class TestJob:
    name: str
    description: str
    tasks: List[str]  # Multiple tasks in one job
    model: str
    backend: str
    tier: str = "core"  # core, extended, edge
    gres: str = "gpu:mi250:1"
    time: str = TIME_SHORT
    extra_args: List[str] = field(default_factory=list)
    checks: List[str] = field(default_factory=list)  # What to verify
    skip_if: str = ""  # Condition to skip (e.g., "no_local_model")


# =============================================================================
# TIER 1: CORE TESTS (Must pass for merge)
# =============================================================================

CORE_TESTS = [
    TestJob(
        name="vllm_1gpu_scoring",
        description="vLLM backend: loglikelihood scoring task",
        tier="core",
        tasks=["arc_easy"],
        model=MODEL_SMALL,
        backend="vllm",
        gres="gpu:mi250:1",
        checks=[
            "output_json_valid",
            "output_path_correct",
            "has_arc_easy_results",
            "gpu_count:1",
            "log_contains:vLLM:",
        ],
    ),
    TestJob(
        name="hf_1gpu_scoring",
        description="HF backend: same task for parity comparison",
        tier="core",
        tasks=["arc_easy"],
        model=MODEL_SMALL,
        backend="hf",
        gres="gpu:mi250:1",
        time=TIME_LONG,
        checks=[
            "output_json_valid",
            "has_arc_easy_results",
            "parity:vllm_1gpu_scoring:arc_easy",
        ],
    ),
    TestJob(
        name="vllm_8gpu_tp",
        description="vLLM 8 GPU tensor parallelism with chat template",
        tier="core",
        tasks=["arc_challenge"],
        model=MODEL_LARGE,
        backend="vllm",
        gres="gpu:mi250:8",
        extra_args=["--apply_chat_template", "--fewshot_as_multiturn"],
        checks=[
            "output_json_valid",
            "gpu_count:8",
            "has_arc_challenge_results",
            "log_contains:tensor_parallel_size=8",
        ],
    ),
    TestJob(
        name="dummy_backend",
        description="Dummy backend (cache-only, no GPU needed)",
        tier="core",
        tasks=["arc_easy"],
        model=MODEL_SMALL,
        backend="dummy",
        gres="gpu:mi250:1",
        extra_args=["--limit", "5"],  # Very small for dummy
        checks=[
            "output_json_valid",
            "has_arc_easy_results",
        ],
    ),
    TestJob(
        name="custom_lm_eval_repo",
        description="Custom lm-eval repo (EleutherAI upstream)",
        tier="core",
        tasks=["arc_easy"],
        model=MODEL_SMALL,
        backend="vllm",
        gres="gpu:mi250:1",
        extra_args=[
            "--lm_eval", "https://github.com/EleutherAI/lm-evaluation-harness@main",
        ],
        checks=[
            "output_json_valid",
            "has_arc_easy_results",
            "log_contains:EleutherAI/lm-evaluation-harness",
        ],
    ),
]

# =============================================================================
# TIER 2: EXTENDED TESTS (Recommended)
# =============================================================================

EXTENDED_TESTS = [
    TestJob(
        name="multilingual_fi",
        description="Multilingual: Finnish HellaSwag",
        tier="extended",
        tasks=["hellaswag_mt_fi"],
        model=MODEL_SMALL,
        backend="vllm",
        gres="gpu:mi250:1",
        checks=[
            "output_json_valid",
            "has_hellaswag_mt_fi_results",
        ],
    ),
    TestJob(
        name="multilingual_sv",
        description="Multilingual: Swedish ARC",
        tier="extended",
        tasks=["arc_challenge_mt_sv"],
        model=MODEL_SMALL,
        backend="vllm",
        gres="gpu:mi250:1",
        checks=[
            "output_json_valid",
            "has_arc_challenge_mt_sv_results",
        ],
    ),
    TestJob(
        name="generation_task",
        description="Generation task: lambada_openai",
        tier="extended",
        tasks=["lambada_openai"],
        model=MODEL_SMALL,
        backend="vllm",
        gres="gpu:mi250:1",
        checks=[
            "output_json_valid",
            "has_lambada_openai_results",
        ],
    ),
    TestJob(
        name="vllm_4gpu_tp",
        description="vLLM 4 GPU intermediate config",
        tier="extended",
        tasks=["arc_easy"],
        model=MODEL_LARGE,
        backend="vllm",
        gres="gpu:mi250:4",
        extra_args=["--apply_chat_template"],
        checks=[
            "output_json_valid",
            "gpu_count:4",
            "has_arc_easy_results",
            "log_contains:tensor_parallel_size=4",
        ],
    ),
    TestJob(
        name="multi_task_single_job",
        description="Multiple evals in single job",
        tier="extended",
        tasks=["arc_easy", "hellaswag", "piqa"],
        model=MODEL_SMALL,
        backend="vllm",
        gres="gpu:mi250:1",
        checks=[
            "output_json_valid",
            "output_path_correct",
            "has_arc_easy_results",
            "has_hellaswag_results",
            "has_piqa_results",
        ],
    ),
    TestJob(
        name="local_model_path",
        description="Local checkpoint path (tests path mapping)",
        tier="extended",
        tasks=["arc_easy"],
        model=MODEL_LOCAL,
        backend="vllm",
        gres="gpu:mi250:4",
        skip_if="no_local_model",
        checks=[
            "output_json_valid",
            "has_arc_easy_results",
            "log_contains:/project",  # Path should be remapped
        ],
    ),
    # Test from PR #7 - fewshot variations
    TestJob(
        name="fewshot_10shot",
        description="10-shot fewshot test (hellaswag default)",
        tier="extended",
        tasks=["hellaswag"],
        model=MODEL_SMALL,
        backend="vllm",
        gres="gpu:mi250:1",
        checks=[
            "output_json_valid",
            "has_hellaswag_results",
        ],
    ),
]

# =============================================================================
# TIER 3: EDGE CASE TESTS (Optional)
# =============================================================================

EDGE_TESTS = [
    TestJob(
        name="custom_model_args",
        description="Custom model_args (max_model_len, dtype)",
        tier="edge",
        tasks=["arc_easy"],
        model=MODEL_SMALL,
        backend="vllm",
        gres="gpu:mi250:1",
        extra_args=[
            "--model_args", "max_model_len=2048,dtype=float16",
        ],
        checks=[
            "output_json_valid",
            "has_arc_easy_results",
            "log_contains:max_model_len=2048",
        ],
    ),
    TestJob(
        name="custom_batch_size",
        description="Custom batch size (8)",
        tier="edge",
        tasks=["arc_easy"],
        model=MODEL_SMALL,
        backend="vllm",
        gres="gpu:mi250:1",
        extra_args=[
            "--batch_size", "8",
        ],
        checks=[
            "output_json_valid",
            "has_arc_easy_results",
        ],
    ),
    TestJob(
        name="trust_remote_code",
        description="Trust remote code flag",
        tier="edge",
        tasks=["arc_easy"],
        model=MODEL_SMALL,
        backend="vllm",
        gres="gpu:mi250:1",
        extra_args=[
            "--trust_remote_code",
        ],
        checks=[
            "output_json_valid",
            "has_arc_easy_results",
        ],
    ),
]

# Combine all tests
TEST_JOBS = CORE_TESTS + EXTENDED_TESTS + EDGE_TESTS


# =============================================================================
# VALIDATION
# =============================================================================

class Validator:
    def __init__(self, output_root: Path, log_dir: Path):
        self.output_root = output_root
        self.log_dir = log_dir
        self.all_results = {}  # Cache for parity checks

    def run_checks(self, job: TestJob, job_id: str) -> list[tuple[str, bool, str]]:
        """Run all checks for a job, return list of (check_name, passed, message)."""
        results = []
        output_dir = self.output_root / job.name

        for check in job.checks:
            passed, msg = self._run_check(check, job, output_dir, job_id)
            results.append((check, passed, msg))

        return results

    def _run_check(self, check: str, job: TestJob, output_dir: Path, job_id: str) -> tuple[bool, str]:
        """Run a single check."""
        log_path = self.log_dir / f"{job_id}.out"

        # Parse check type
        if check == "output_json_valid":
            return self._check_json_valid(output_dir, job.tasks)

        elif check.startswith("has_") and check.endswith("_results"):
            task = check[4:-8]  # Extract task name
            return self._check_has_task_results(output_dir, task)

        elif check.startswith("gpu_count:"):
            expected = int(check.split(":")[1])
            return self._check_gpu_count(log_path, expected)

        elif check.startswith("parity:"):
            # Format: parity:other_job:task
            parts = check.split(":")
            other_job = parts[1]
            task = parts[2] if len(parts) > 2 else job.tasks[0]
            return self._check_parity(output_dir, other_job, [task])

        elif check.startswith("log_contains:"):
            pattern = check.split(":", 1)[1]
            return self._check_log_contains(log_path, pattern)

        elif check == "output_path_correct":
            return self._check_output_path(output_dir, job)

        return False, f"Unknown check: {check}"

    def _check_output_path(self, output_dir: Path, job: TestJob) -> tuple[bool, str]:
        """Check that output files are written to the expected directory."""
        if not output_dir.exists():
            return False, f"Output dir does not exist: {output_dir}"

        # Find JSON files in output dir
        json_files = list(output_dir.glob("*.json"))
        if not json_files:
            return False, f"No JSON files in {output_dir}"

        # Verify each task has an output file
        missing = []
        for task in job.tasks:
            # Check for both vllm_ prefixed and unprefixed files
            found = any(
                task in f.stem for f in json_files
            )
            if not found:
                missing.append(task)

        if missing:
            return False, f"Missing output for tasks: {missing}"

        return True, f"Output files in correct path: {output_dir.name}/ ({len(json_files)} files)"

    def _check_log_contains(self, log_path: Path, pattern: str) -> tuple[bool, str]:
        """Check that log file contains a pattern."""
        if not log_path.exists():
            return False, f"Log not found: {log_path}"

        content = log_path.read_text()
        if pattern in content:
            return True, f"Found '{pattern}' in log"
        return False, f"Pattern '{pattern}' not found in log"

    def _check_json_valid(self, output_dir: Path, tasks: list[str]) -> tuple[bool, str]:
        """Check that valid JSON output exists for each task."""
        for task in tasks:
            # Find JSON files containing the task name (handles timestamps in filenames)
            # e.g., vllm_arc_easy_2026-01-27T13-52-57.json or arc_easy.json
            matching_files = list(output_dir.glob(f"*{task}*.json"))
            # Exclude samples files (jsonl files that got renamed)
            matching_files = [f for f in matching_files if "samples_" not in f.name]

            found = False
            for path in matching_files:
                try:
                    with open(path) as f:
                        data = json.load(f)
                    if "results" in data:
                        found = True
                        self.all_results[f"{output_dir.name}:{task}"] = data
                        break
                except json.JSONDecodeError:
                    return False, f"Invalid JSON in {path}"
            if not found:
                return False, f"No valid output for task {task}"
        return True, f"All {len(tasks)} task outputs valid"

    def _check_has_task_results(self, output_dir: Path, task: str) -> tuple[bool, str]:
        """Check that results contain data for a specific task."""
        key = f"{output_dir.name}:{task}"
        if key not in self.all_results:
            # Try to load it
            for pattern in [f"{task}.json", f"vllm_{task}.json"]:
                path = output_dir / pattern
                if path.exists():
                    with open(path) as f:
                        self.all_results[key] = json.load(f)
                    break

        if key not in self.all_results:
            return False, f"No results loaded for {task}"

        data = self.all_results[key]
        results = data.get("results", {})

        # Check if any result key contains the task name
        for result_key in results:
            if task in result_key or task.replace("_", "") in result_key.replace("_", ""):
                return True, f"Found results for {task}"

        return False, f"No results matching {task} in output"

    def _check_gpu_count(self, log_path: Path, expected: int) -> tuple[bool, str]:
        """Verify GPU count from logs."""
        if not log_path.exists():
            return False, f"Log not found: {log_path}"

        content = log_path.read_text()

        # Look for "Devices: N" from PyTorch check
        match = re.search(r"Devices:\s*(\d+)", content)
        if match:
            actual = int(match.group(1))
            if actual >= expected:
                return True, f"GPU count OK: {actual}"
            return False, f"GPU count: expected {expected}, got {actual}"

        # Fallback: look for "GPUs: N"
        match = re.search(r"GPUs:\s*(\d+)", content)
        if match:
            actual = int(match.group(1))
            if actual >= expected:
                return True, f"GPU count OK: {actual}"
            return False, f"GPU count: expected {expected}, got {actual}"

        return False, "Could not determine GPU count"

    def _check_parity(self, output_dir: Path, other_job: str, tasks: list[str]) -> tuple[bool, str]:
        """Compare results with another job for parity."""
        diffs = []
        for task in tasks:
            key1 = f"{output_dir.name}:{task}"
            key2 = f"{other_job}:{task}"

            if key1 not in self.all_results or key2 not in self.all_results:
                return False, f"Missing results for parity check: {key1} or {key2}"

            # Extract a comparable metric
            r1 = self.all_results[key1].get("results", {})
            r2 = self.all_results[key2].get("results", {})

            # Find common metrics
            for task_key in r1:
                if task in task_key:
                    m1 = r1[task_key]
                    m2 = r2.get(task_key, {})

                    # Compare acc or acc_norm
                    for metric in ["acc,none", "acc_norm,none", "acc", "acc_norm"]:
                        if metric in m1 and metric in m2:
                            v1, v2 = m1[metric], m2[metric]
                            diff = abs(v1 - v2)
                            diffs.append((task, metric, v1, v2, diff))
                            break

        if not diffs:
            return False, "No comparable metrics found"

        # Check all diffs are within tolerance (2%)
        max_diff = max(d[4] for d in diffs)
        if max_diff < 0.02:
            summary = ", ".join(f"{d[0]}:{d[4]:.4f}" for d in diffs)
            return True, f"Parity OK (max diff {max_diff:.4f}): {summary}"
        else:
            summary = ", ".join(f"{d[0]}:{d[2]:.3f}vs{d[3]:.3f}" for d in diffs)
            return False, f"Parity FAIL (diff {max_diff:.4f}): {summary}"


# =============================================================================
# TEST RUNNER
# =============================================================================

class TestRunner:
    def __init__(self, output_root: str, dry_run: bool = False):
        self.output_root = Path(output_root)
        self.dry_run = dry_run
        self.results_file = self.output_root / "test_results.json"
        self.log_dir = self.output_root / "logs"

    def setup(self):
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def should_skip(self, job: TestJob) -> tuple[bool, str]:
        """Check if job should be skipped."""
        if job.skip_if == "no_local_model" and not job.model:
            return True, "MODEL_LOCAL not configured"
        return False, ""

    def run_job(self, job: TestJob) -> dict:
        """Submit a test job (may include multiple tasks)."""
        skip, reason = self.should_skip(job)
        if skip:
            return {"name": job.name, "status": "skipped", "reason": reason}

        output_dir = self.output_root / job.name

        # Build command - run all tasks in one job
        cmd = [
            "python", "main.py",
        ] + job.tasks + [
            "--model", job.model,
            "--backend", job.backend,
            "--partition", PARTITION,
            "--time", job.time,
            "--gres", job.gres,
            "--limit", str(LIMIT),
            "--output_dir", str(output_dir),
            "--log_dir", str(self.log_dir),
            "--force",
        ] + job.extra_args

        print(f"\n{'='*60}")
        print(f"JOB: {job.name}")
        print(f"     {job.description}")
        print(f"     Tasks: {', '.join(job.tasks)}")
        print(f"     {job.gres}, {job.time}")
        print(f"CMD: {' '.join(cmd)}")
        print(f"{'='*60}")

        if self.dry_run:
            return {"name": job.name, "status": "dry-run", "cmd": " ".join(cmd)}

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            # Parse job IDs (one per task)
            job_ids = []
            for line in result.stdout.split("\n"):
                match = re.search(r'"job_id":\s*"(\d+)"', line)
                if match:
                    job_ids.append(match.group(1))

            return {
                "name": job.name,
                "status": "submitted" if result.returncode == 0 else "failed",
                "job_ids": job_ids,
                "tasks": job.tasks,
                "checks": job.checks,
                "output_dir": str(output_dir),
                "returncode": result.returncode,
                "stdout": result.stdout[-2000:],  # Last 2000 chars
                "stderr": result.stderr[-1000:] if result.stderr else "",
            }
        except subprocess.TimeoutExpired:
            return {"name": job.name, "status": "timeout"}
        except Exception as e:
            return {"name": job.name, "status": "error", "error": str(e)}

    def run_all(self, jobs: list[TestJob]) -> list[dict]:
        """Run all test jobs."""
        self.setup()
        results = []

        for job in jobs:
            result = self.run_job(job)
            results.append(result)

        with open(self.results_file, "w") as f:
            json.dump(results, f, indent=2)

        self._print_summary(results)
        return results

    def check_results(self, jobs: list[TestJob]) -> list[dict]:
        """Check results of previously submitted jobs."""
        if not self.results_file.exists():
            print("No results file. Run --run first.")
            return []

        with open(self.results_file) as f:
            results = json.load(f)

        validator = Validator(self.output_root, self.log_dir)
        job_map = {j.name: j for j in jobs}

        for r in results:
            if r.get("status") != "submitted":
                continue

            # Check SLURM job status
            job_ids = r.get("job_ids", [])
            if job_ids:
                statuses = [self._get_job_status(jid) for jid in job_ids]
                r["job_statuses"] = statuses

                # All completed?
                if all(s == "COMPLETED" for s in statuses):
                    r["job_status"] = "COMPLETED"
                elif any(s == "FAILED" for s in statuses):
                    r["job_status"] = "FAILED"
                elif any(s in ("PENDING", "RUNNING") for s in statuses):
                    r["job_status"] = "RUNNING"
                else:
                    r["job_status"] = "UNKNOWN"

            # Run validation if completed
            if r.get("job_status") == "COMPLETED":
                job = job_map.get(r["name"])
                if job:
                    checks = validator.run_checks(job, job_ids[0] if job_ids else "")
                    r["validation"] = [
                        {"check": c, "passed": p, "message": m}
                        for c, p, m in checks
                    ]
                    r["all_passed"] = all(p for _, p, _ in checks)

        with open(self.results_file, "w") as f:
            json.dump(results, f, indent=2)

        self._print_check_summary(results)
        return results

    def _get_job_status(self, job_id: str) -> str:
        try:
            result = subprocess.run(
                ["sacct", "-j", job_id, "--format=State", "--noheader", "-P"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                states = result.stdout.strip().split("\n")
                return states[0].strip() if states else "UNKNOWN"
        except:
            pass
        return "UNKNOWN"

    def _print_summary(self, results: list[dict]):
        print(f"\n{'='*60}")
        print("SUBMITTED JOBS")
        print(f"{'='*60}")
        for r in results:
            status = r.get("status", "unknown")
            if status == "submitted":
                ids = ",".join(r.get("job_ids", []))
                print(f"  OK  {r['name']}: jobs {ids}")
            elif status == "skipped":
                print(f"  --  {r['name']}: {r.get('reason', 'skipped')}")
            elif status == "dry-run":
                print(f"  >>  {r['name']}: {r.get('cmd', '')[:60]}...")
            else:
                print(f"  !!  {r['name']}: {status}")

    def _print_check_summary(self, results: list[dict]):
        print(f"\n{'='*60}")
        print("VALIDATION RESULTS")
        print(f"{'='*60}")
        for r in results:
            name = r["name"]
            status = r.get("job_status", r.get("status", "unknown"))

            if status == "skipped":
                print(f"  --  {name}: skipped")
            elif status != "COMPLETED":
                print(f"  ..  {name}: {status}")
            else:
                validations = r.get("validation", [])
                all_passed = r.get("all_passed", False)
                icon = "OK" if all_passed else "!!"
                print(f"  {icon}  {name}:")
                for v in validations:
                    mark = "+" if v["passed"] else "X"
                    print(f"      [{mark}] {v['check']}: {v['message']}")

    def clean(self):
        import shutil
        if self.output_root.exists():
            shutil.rmtree(self.output_root)
            print(f"Cleaned: {self.output_root}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Test container migration")
    parser.add_argument("--dry-run", action="store_true", help="Show commands without running")
    parser.add_argument("--run", action="store_true", help="Submit test jobs")
    parser.add_argument("--check", action="store_true", help="Check job results")
    parser.add_argument("--clean", action="store_true", help="Clean test outputs")
    parser.add_argument("--parity", action="store_true", help="Check backend parity")
    parser.add_argument("--output", default=OUTPUT_ROOT, help="Output directory")
    parser.add_argument("--filter", help="Only run tests matching this pattern")
    parser.add_argument("--tier", choices=["core", "extended", "edge", "all"],
                        default="all", help="Test tier to run (default: all)")
    parser.add_argument("--list", action="store_true", help="List all available tests")

    args = parser.parse_args()

    if args.list:
        print("\n=== CORE TESTS (required for merge) ===")
        for j in CORE_TESTS:
            print(f"  {j.name}: {j.description}")
        print("\n=== EXTENDED TESTS (recommended) ===")
        for j in EXTENDED_TESTS:
            print(f"  {j.name}: {j.description}")
        print("\n=== EDGE CASE TESTS (optional) ===")
        for j in EDGE_TESTS:
            print(f"  {j.name}: {j.description}")
        print(f"\nTotal: {len(TEST_JOBS)} tests")
        return

    if not any([args.dry_run, args.run, args.check, args.clean]):
        parser.print_help()
        return

    runner = TestRunner(args.output, dry_run=args.dry_run)

    # Filter by tier
    jobs = TEST_JOBS
    if args.tier != "all":
        if args.tier == "core":
            jobs = CORE_TESTS
        elif args.tier == "extended":
            jobs = CORE_TESTS + EXTENDED_TESTS
        elif args.tier == "edge":
            jobs = TEST_JOBS  # All tiers
        print(f"Running tier '{args.tier}': {len(jobs)} jobs")

    # Filter by name pattern
    if args.filter:
        jobs = [j for j in jobs if args.filter in j.name]
        print(f"Filtered to {len(jobs)} jobs matching '{args.filter}'")

    if args.clean:
        runner.clean()
    elif args.run or args.dry_run:
        runner.run_all(jobs)
    elif args.check:
        runner.check_results(jobs)


if __name__ == "__main__":
    main()
