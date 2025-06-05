import unittest
from unittest.mock import patch, MagicMock
import sys
import os
from pathlib import Path
import yaml

# Add the project root to the Python path to allow importing main_pipeline
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from main_pipeline import main, load_config, generate_pipeline_report
from src.data_flow_manager import DataFlowManager
from src.model_manager import ModelManager
from src.progress_tracker import ProgressTracker
from integration_interfaces import MILStep, SSLStep, XAIStep, FusionStep

class TestMainPipeline(unittest.TestCase):
    def setUp(self):
        # This method will be called before each test
        self.test_config_path = project_root / "tests" / "config" / "test_pipeline_config.yaml"
        self.test_output_dir = project_root / "tests" / "outputs"

        # Ensure the test output directory exists
        self.test_output_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        # This method will be called after each test
        # Clean up created files if necessary
        report_path = self.test_output_dir / 'pipeline_report.yaml'
        if report_path.exists():
            os.remove(report_path)

        progress_file = self.test_output_dir / "pipeline_progress.json"
        if progress_file.exists():
            os.remove(progress_file)

    @patch('main_pipeline.FusionStep')
    @patch('main_pipeline.XAIStep')
    @patch('main_pipeline.SSLStep')
    @patch('main_pipeline.MILStep')
    @patch('main_pipeline.DataFlowManager')
    @patch('main_pipeline.ModelManager')
    @patch('main_pipeline.ProgressTracker')
    @patch('main_pipeline.generate_pipeline_report')
    def test_pipeline_runs_all_steps_successfully(self, mock_generate_report, mock_progress_tracker, mock_model_manager, mock_data_manager, mock_mil_step, mock_ssl_step, mock_xai_step, mock_fusion_step):
        # Configure mocks
        mock_mil_step.return_value.execute.return_value = {"status": "MIL complete"}
        mock_ssl_step.return_value.execute.return_value = {"status": "SSL complete"}
        mock_xai_step.return_value.execute.return_value = {"status": "XAI complete"}
        mock_fusion_step.return_value.execute.return_value = {"status": "Fusion complete"}

        # Construct arguments for main()
        sys.argv = ['main_pipeline.py', '--config', str(self.test_config_path)]

        main()

        # Assert DataFlowManager, ModelManager, ProgressTracker were initialized
        mock_data_manager.assert_called_once()
        mock_model_manager.assert_called_once_with(str(project_root / "tests" / "models" / "")) # Path from test_config
        mock_progress_tracker.assert_called_once_with(str(project_root / "tests" / "outputs" / "")) # Path from test_config

        # Assert that each step was initialized and execute was called
        mock_mil_step.assert_called_once()
        mock_mil_step.return_value.execute.assert_called_once()

        mock_ssl_step.assert_called_once()
        mock_ssl_step.return_value.execute.assert_called_once()

        mock_xai_step.assert_called_once()
        mock_xai_step.return_value.execute.assert_called_once()

        mock_fusion_step.assert_called_once()
        mock_fusion_step.return_value.execute.assert_called_once()

        # Assert generate_pipeline_report was called
        expected_results = {
            'mil': {"status": "MIL complete"},
            'ssl': {"status": "SSL complete"},
            'xai': {"status": "XAI complete"},
            'fusion': {"status": "Fusion complete"}
        }
        mock_generate_report.assert_called_once_with(expected_results, str(self.test_output_dir / ""))

        # Check for pipeline_progress.json (created by ProgressTracker)
        # We can't directly check its content without more complex mocking of ProgressTracker's methods
        # For now, we assume if ProgressTracker was called, it behaved as expected.
        # A more advanced test could mock file I/O for ProgressTracker.

    @patch('main_pipeline.FusionStep')
    @patch('main_pipeline.XAIStep')
    @patch('main_pipeline.SSLStep')
    @patch('main_pipeline.MILStep')
    @patch('main_pipeline.generate_pipeline_report')
    def test_pipeline_runs_specific_steps(self, mock_generate_report, mock_mil_step, mock_ssl_step, mock_xai_step, mock_fusion_step):
        # Configure mocks for execute methods
        mock_mil_step.return_value.execute.return_value = {"status": "MIL complete"}
        mock_ssl_step.return_value.execute.return_value = {"status": "SSL complete"}

        # Construct arguments for main() to run only mil and ssl steps
        sys.argv = ['main_pipeline.py', '--config', str(self.test_config_path), '--steps', 'mil', 'ssl']

        main()

        # Assert that only MIL and SSL steps were initialized and executed
        mock_mil_step.assert_called_once()
        mock_mil_step.return_value.execute.assert_called_once()

        mock_ssl_step.assert_called_once()
        mock_ssl_step.return_value.execute.assert_called_once()

        # Assert that XAI and Fusion steps were initialized (due to current main_pipeline structure)
        # but their execute methods were not called
        mock_xai_step.assert_called_once() # Constructor is called
        mock_xai_step.return_value.execute.assert_not_called()
        mock_fusion_step.assert_called_once() # Constructor is called
        mock_fusion_step.return_value.execute.assert_not_called()

        # Assert generate_pipeline_report was called with results from mil and ssl only
        expected_results = {
            'mil': {"status": "MIL complete"},
            'ssl': {"status": "SSL complete"}
        }
        # Ensure config is passed correctly to DataFlowManager, ModelManager, ProgressTracker (implicitly tested by previous test)
        # We are focusing on step selection logic here
        mock_generate_report.assert_called_once_with(expected_results, str(self.test_output_dir / ""))

    def test_load_config(self):
        config = load_config(str(self.test_config_path))
        self.assertIsNotNone(config)
        self.assertEqual(config['pipeline']['name'], "GRAPHITE_Pipeline_Test")
        self.assertIn('paths', config)
        self.assertIn('step_1_mil', config)

    def test_generate_pipeline_report(self):
        sample_results = {"step1": "success", "step2": "failure"}
        report_file_path = self.test_output_dir / 'pipeline_report.yaml'

        # Ensure the file does not exist before the test
        if report_file_path.exists():
            os.remove(report_file_path)

        generate_pipeline_report(sample_results, str(self.test_output_dir))

        self.assertTrue(report_file_path.exists())

        with open(report_file_path, 'r') as f:
            report_data = yaml.safe_load(f)

        self.assertEqual(report_data, sample_results)

        # Clean up the created file
        os.remove(report_file_path)

    def test_pipeline_handles_invalid_config_path(self):
        # Construct arguments for main() with a non-existent config file
        invalid_config_path = project_root / "tests" / "config" / "non_existent_config.yaml"
        sys.argv = ['main_pipeline.py', '--config', str(invalid_config_path)]

        with self.assertRaises(FileNotFoundError):
            main()

    @patch('main_pipeline.FusionStep')
    @patch('main_pipeline.XAIStep')
    @patch('main_pipeline.SSLStep')
    @patch('main_pipeline.MILStep')
    @patch('main_pipeline.generate_pipeline_report') # Mock to check if it's called
    def test_pipeline_handles_failing_step(self, mock_generate_report, mock_mil_step, mock_ssl_step, mock_xai_step, mock_fusion_step):
        # Configure MILStep's execute to raise an exception
        mock_mil_step.return_value.execute.side_effect = Exception("MIL Step Failed")

        # Configure other steps to behave normally (though they shouldn't be called if MIL fails early)
        mock_ssl_step.return_value.execute.return_value = {"status": "SSL complete"}
        mock_xai_step.return_value.execute.return_value = {"status": "XAI complete"}
        mock_fusion_step.return_value.execute.return_value = {"status": "Fusion complete"}

        # Construct arguments for main()
        sys.argv = ['main_pipeline.py', '--config', str(self.test_config_path)]

        # Expect the main() function to catch the exception from the step and exit,
        # or propagate it if not handled within main().
        # Based on current main_pipeline.py, it seems exceptions from steps are not caught within main,
        # so the exception should propagate.
        with self.assertRaisesRegex(Exception, "MIL Step Failed"):
            main()

        # Assert that MIL step was called
        mock_mil_step.assert_called_once()
        mock_mil_step.return_value.execute.assert_called_once()

        # Assert that subsequent steps were not called due to the failure in MIL
        mock_ssl_step.return_value.execute.assert_not_called()
        mock_xai_step.return_value.execute.assert_not_called()
        mock_fusion_step.return_value.execute.assert_not_called()

        # Assert that generate_pipeline_report is NOT called because the pipeline failed mid-way
        # and the current main_pipeline.py structure doesn't seem to generate a partial report on exception.
        # If it were to generate a partial report, this assertion would change.
        mock_generate_report.assert_not_called()


if __name__ == '__main__':
    unittest.main()
