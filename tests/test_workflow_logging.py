import io
import sys
import tempfile
import unittest
from pathlib import Path

from rich.console import Console
from rich.file_proxy import FileProxy

from workflow_sym import DualLogger


class WorkflowLoggingTests(unittest.TestCase):
    def test_dual_logger_unwraps_rich_stdout_proxy(self):
        original_stdout = sys.stdout
        terminal = io.StringIO()
        logger = None

        try:
            sys.stdout = terminal
            console = Console()
            rich_proxy = FileProxy(console, terminal)
            sys.stdout = rich_proxy

            with tempfile.TemporaryDirectory() as temp_dir:
                log_path = Path(temp_dir) / "workflow.log"
                logger = DualLogger(log_path)
                sys.stdout = logger
                print("workflow started")
                logger.close()
                logger = None

                self.assertEqual(logger_text := log_path.read_text(encoding="utf-8"), "workflow started\n")
                self.assertIn("workflow started", terminal.getvalue())
                self.assertEqual(logger_text.count("workflow started"), 1)
        finally:
            sys.stdout = original_stdout
            if logger is not None:
                logger.close()


if __name__ == "__main__":
    unittest.main()
