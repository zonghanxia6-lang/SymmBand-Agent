import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from band_workflow import BandWorkflowConfig, check_band_environment, run_band_workflow


class BandWorkflowTests(unittest.TestCase):
    def test_environment_check_reports_ready_runtime(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runner = root / "agent_runner.py"
            python = root / "python.exe"
            runner.touch()
            python.touch()
            completed = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout=json.dumps({"ready": True, "errors": [], "irvsp": "irvsp"}),
            )
            with patch("band_workflow.subprocess.run", return_value=completed):
                result = check_band_environment(root, python)

        self.assertTrue(result["ready"])

    def test_subprocess_report_is_converted_to_result(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            analyzer_root = root / "analyzer"
            analyzer_root.mkdir()
            (analyzer_root / "agent_runner.py").touch()
            python = root / "python.exe"
            python.touch()
            poscar = root / "POSCAR_NaBi"
            poscar.write_text("test", encoding="utf-8")
            output_root = root / "band_analysis"
            output_root.mkdir()
            band_image = output_root / "bands" / "band_001_NaBi.png"
            report = {
                "requested_count": 1,
                "completed_count": 1,
                "failed_count": 0,
                "bands_directory": str(band_image.parent),
                "band_images": [str(band_image)],
                "results": [{"status": "completed"}],
            }
            (output_root / "band_report.json").write_text(
                json.dumps(report), encoding="utf-8"
            )
            completed = subprocess.CompletedProcess(args=[], returncode=0)
            config = BandWorkflowConfig(
                structure_paths=[poscar],
                output_root=output_root,
                analyzer_root=analyzer_root,
                python_executable=python,
            )

            with patch("band_workflow.subprocess.run", return_value=completed) as run_mock:
                result = run_band_workflow(config)

        self.assertEqual(result.completed_count, 1)
        self.assertEqual(result.band_images, [str(band_image.resolve())])
        command = run_mock.call_args.args[0]
        self.assertNotIn("shell", run_mock.call_args.kwargs)
        self.assertIn("--manifest", command)
        self.assertIn("--report", command)


if __name__ == "__main__":
    unittest.main()
