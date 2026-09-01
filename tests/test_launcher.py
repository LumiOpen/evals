import argparse
import contextlib
import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import main
from evals.harnesses import LMEvalHarness


class SubmitSlurmTests(unittest.TestCase):
    @mock.patch("main.subprocess.run")
    def test_parses_parsable_job_id(self, run):
        run.return_value = SimpleNamespace(
            returncode=0,
            stdout="12345;lumi\n",
            stderr="",
        )

        self.assertEqual(main.submit_slurm("job.sbatch"), "12345")
        run.assert_called_once_with(
            ["sbatch", "--parsable", "job.sbatch"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )

    @mock.patch("main.subprocess.run")
    def test_rejects_failed_submission(self, run):
        run.return_value = SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="accounting policy rejected job\n",
        )

        with self.assertRaisesRegex(RuntimeError, "accounting policy rejected job"):
            main.submit_slurm("job.sbatch")

    @mock.patch("main.subprocess.run")
    def test_rejects_unexpected_submission_output(self, run):
        run.return_value = SimpleNamespace(
            returncode=0,
            stdout="Submitted batch job 12345\n",
            stderr="",
        )

        with self.assertRaisesRegex(RuntimeError, "unexpected sbatch"):
            main.submit_slurm("job.sbatch")


class LMEvalTemplateTests(unittest.TestCase):
    def render(self, *, forward_hf_token=False, extras=(), model="Qwen/Qwen2.5-0.5B-Instruct"):
        slurm_config = {
            "name": "smoke",
            "account": "project_462001516",
            "partition": "dev-g",
            "gres": "gpu:mi250:1",
            "time": "00:10:00",
            "log_dir": "/scratch/logs",
            "dependency": None,
            "mem": "60G",
            "cpus_per_task": 7,
        }
        env_vars = {
            "MODEL": model,
            "OUTPUT_FILE": "/scratch/results/result.json",
            "LM_EVAL_PATH": "",
            "LM_EVAL_REPO": "https://github.com/LumiOpen/lm-evaluation-harness",
            "LM_EVAL_REF": "d77817d967c9a307b2d5f6e78a5a6875b79f6b12",
            "TRANSFORMERS_VERSION": "4.57.1",
            "EXTRA_PIP_PACKAGES": list(extras),
            "CONTAINER": "/appl/local/image.sif",
            "FORWARD_HF_TOKEN": forward_hf_token,
            "APPLY_CHAT_TEMPLATE": False,
            "FEWSHOT_AS_MULTITURN": False,
            "BACKEND": "dummy",
            "MODEL_ARGS": "",
            "BATCH_SIZE": "",
            "LIMIT": "1",
            "LM_EVAL_ARGS": "",
        }
        return LMEvalHarness(["arc_easy"]).generate_script(
            slurm_config,
            env_vars,
            backend="dummy",
        )

    def test_renders_isolated_runtime_and_pinned_fetch(self):
        script = self.render(extras=main.extra_packages_for_eval("ifeval"))

        self.assertIn("#SBATCH --mem=60G", script)
        self.assertIn("#SBATCH --cpus-per-task=7", script)
        self.assertIn('JOB_TMP="${TMPDIR:-/tmp}/lumi-evals-${SLURM_JOB_ID}"', script)
        self.assertIn('PIP_TARGET="$JOB_TMP/pip-packages"', script)
        self.assertIn('EVAL_HARNESS_DIR="$JOB_TMP/lm-eval"', script)
        self.assertIn('git -C "$EVAL_HARNESS_DIR" fetch -q --depth=1 origin "$REPO_REF"', script)
        self.assertNotIn("git clone --depth 1 -b", script)
        self.assertNotIn("find /project/cache/huggingface/hub/.locks", script)
        self.assertIn('"langdetect"', script)
        self.assertIn('"immutabledict"', script)
        self.assertIn('"nltk>=3.9.1"', script)

    def test_token_forwarding_is_opt_in(self):
        without_token = self.render(forward_hf_token=False)
        with_token = self.render(forward_hf_token=True)

        self.assertNotIn('HF_TOKEN=${HF_TOKEN:-}', without_token)
        self.assertIn('HF_TOKEN=${HF_TOKEN:-}', with_token)

    def test_model_value_is_shell_quoted(self):
        script = self.render(model="model$(touch /tmp/not-allowed)")

        self.assertIn("export MODEL='model$(touch /tmp/not-allowed)'", script)

    def test_rendered_script_has_valid_bash_syntax(self):
        script = self.render()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh") as file:
            file.write(script)
            file.flush()
            result = subprocess.run(
                ["bash", "-n", file.name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )
        self.assertEqual(result.returncode, 0, result.stderr)


class RunEvalReceiptTests(unittest.TestCase):
    def test_returns_and_records_structured_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(
                output_dir=str(root / "results"),
                output_root=str(root / "unused"),
                model="dummy/model",
                tokenizer="dummy/model",
                backend="dummy",
                lm_eval="https://github.com/LumiOpen/lm-evaluation-harness@deadbeef",
                work_dir=str(root / "work"),
                log_dir=str(root / "logs"),
                trust_remote_code=False,
                apply_chat_template=False,
                fewshot_as_multiturn=False,
                model_args="",
                lm_eval_args="",
                limit=1,
                batch_size="",
                container="/appl/local/image.sif",
                forward_hf_token=False,
                project="project_462001516",
                partition="dev-g",
                gres="gpu:mi250:1",
                time="00:10:00",
                dependency=None,
                mem="60G",
                cpus_per_task=7,
                comment="test",
                dryrun=False,
                force=False,
            )
            previous = os.getcwd()
            os.chdir(root)
            try:
                with mock.patch("main.submit_slurm", return_value="12345"):
                    with contextlib.redirect_stdout(io.StringIO()):
                        receipt = main.run_eval("arc_easy", args)
            finally:
                os.chdir(previous)

            self.assertEqual(receipt["status"], "submitted")
            self.assertEqual(receipt["job_id"], "12345")
            self.assertEqual(receipt["mem"], "60G")
            self.assertEqual(receipt["cpus_per_task"], 7)
            self.assertEqual(receipt["lm_eval_ref"], "deadbeef")
            self.assertEqual(Path(receipt["script_name"]).parent, root / "work")
            history = json.loads((root / "command_history.jsonl").read_text())
            self.assertEqual(history, receipt)
            script_path = Path(receipt["script_name"])
            if script_path.exists():
                script_path.unlink()


class TaskDependencyTests(unittest.TestCase):
    def test_ifeval_variants_get_required_dependencies(self):
        expected = ["langdetect", "immutabledict", "nltk>=3.9.1"]
        self.assertEqual(main.extra_packages_for_eval("ifeval"), expected)
        self.assertEqual(main.extra_packages_for_eval("ifeval_fi"), expected)
        self.assertEqual(main.extra_packages_for_eval("arc_easy"), [])


class ResourceValidationTests(unittest.TestCase):
    def test_accepts_lumi_resource_values(self):
        self.assertEqual(main.parse_project("project_462001516"), "project_462001516")
        self.assertEqual(main.parse_partition("standard-g"), "standard-g")
        self.assertEqual(main.parse_gres("gpu:mi250:8"), "gpu:mi250:8")
        self.assertEqual(main.parse_gres("gpu:1"), "gpu:1")
        self.assertEqual(main.parse_memory("60g"), "60G")
        self.assertEqual(main.parse_positive_int("7"), 7)
        self.assertEqual(main.parse_time_limit("48:00:00"), "48:00:00")

    def test_rejects_injected_or_invalid_resource_values(self):
        invalid_values = [
            (main.parse_project, "project_462001516\n#SBATCH --account=other"),
            (main.parse_partition, "dev-g;id"),
            (main.parse_gres, "gpu:mi250:9"),
            (main.parse_memory, "all"),
            (main.parse_positive_int, "0"),
            (main.parse_time_limit, "10 minutes"),
        ]
        for parser, value in invalid_values:
            with self.subTest(parser=parser.__name__, value=value):
                with self.assertRaises((ValueError, argparse.ArgumentTypeError)):
                    parser(value)


if __name__ == "__main__":
    unittest.main()
