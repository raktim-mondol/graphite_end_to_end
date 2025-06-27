from typing import Dict, Any
import logging
import importlib
import sys
from contextlib import contextmanager
from src.data_flow_manager import DataFlowManager
from src.model_manager import ModelManager
from src.progress_tracker import ProgressTracker

logger = logging.getLogger(__name__)

@contextmanager
def _patched_argv(args):
    """
    Temporarily patch ``sys.argv`` so that argparse-based CLI
    programs can be invoked programmatically.
    """
    original_argv = sys.argv[:]
    # Argparse expects argv[0] to be the program name.
    sys.argv = ["pipeline_integration"] + args
    try:
        yield
    finally:
        sys.argv = original_argv


def _config_to_cli_args(config_dict: Dict[str, Any]) -> list:
    """
    Convert a configuration dictionary into a list of
    command-line style arguments:  {'epochs': 10, 'verbose': True}
    -> ['--epochs', '10', '--verbose']
    Boolean flags that are False are omitted.
    """
    cli_args: list[str] = []
    for key, value in config_dict.items():
        # --------------------------------------------------------------
        # Skip parameters that are explicitly unset / null in YAML
        # --------------------------------------------------------------
        if value is None:
            continue

        flag = f"--{key}"

        # --------------------  Boolean flags  ------------------------
        if isinstance(value, bool):
            if value:            # include only when True
                cli_args.append(flag)
            continue

        # --------------------  List / Tuple  -------------------------
        if isinstance(value, (list, tuple)):
            # filter out None values inside the sequence
            cleaned = [v for v in value if v is not None]
            if not cleaned:      # skip if list is empty after cleaning
                continue
            cli_args.append(flag)
            cli_args.extend(map(str, cleaned))
            continue

        # --------------------  Scalar values  ------------------------
        cli_args.extend([flag, str(value)])
    return cli_args


class StepInterface:
    """Base interface for all pipeline steps."""

    def __init__(self, config: Dict, data_manager: DataFlowManager,
                 model_manager: ModelManager, progress_tracker: ProgressTracker):
        self.config = config
        self.data_manager = data_manager
        self.model_manager = model_manager
        self.progress_tracker = progress_tracker

    def execute(self) -> Dict[str, Any]:
        raise NotImplementedError

class MILStep(StepInterface):
    def execute(self) -> Dict[str, Any]:
        self.progress_tracker.start_step("mil", "Training MIL model")
        try:
            # Map pipeline config to MIL script parameters
            mil_config = self.config.get("step_1_mil", {})
            
            # Convert to the format expected by the MIL script
            mil_args = {
                "num_epochs": mil_config.get("epochs", 50),
                "batch_size": mil_config.get("batch_size", 16),
                "learning_rate": mil_config.get("learning_rate", 0.001),
                "early_stopping_patience": mil_config.get("early_stopping_patience", 10)
            }
            
            # Option 1: Call train.py directly
            # mil_module = importlib.import_module("training_step_1.mil_classification.train")
            
            # Option 2: Use the launcher script (preferred as it handles dependencies)
            mil_module = importlib.import_module("training_step_1.run_training")
            
            # Convert config to CLI args and patch sys.argv
            cli_args = _config_to_cli_args(mil_args)
            with _patched_argv(cli_args):
                results = mil_module.main()
        except Exception as exc:  # pragma: no cover
            logger.exception("MIL step failed: %s", exc)
            self.progress_tracker.complete_step("mil", {"error": str(exc)})
            raise

        # Ensure results is a dictionary
        if results is None:
            results = {}
        if isinstance(results, dict) and 'model' in results:
            self.model_manager.save_model(results['model'], 'mil_model', 'step1',
                                          {'accuracy': results.get('accuracy')})
        self.data_manager.save_step_output('step1_mil', results if isinstance(results, dict) else {})
        self.progress_tracker.complete_step("mil", results if isinstance(results, dict) else {})
        return results if isinstance(results, dict) else {}

class SSLStep(StepInterface):
    def execute(self) -> Dict[str, Any]:
        self.progress_tracker.start_step("ssl", "Training HierGAT model")
        try:
            ssl_module = importlib.import_module(
                "training_step_2.self_supervised_training.train"
            )
            cli_args = _config_to_cli_args(self.config.get("step_2_ssl", {}))
            with _patched_argv(cli_args):
                results = ssl_module.main()
        except Exception as exc:  # pragma: no cover
            logger.exception("SSL step failed: %s", exc)
            self.progress_tracker.complete_step("ssl", {"error": str(exc)})
            raise

        if results is None:
            results = {}
        if isinstance(results, dict) and 'model' in results:
            self.model_manager.save_model(results['model'], 'hiergat_model', 'step2')
        self.data_manager.save_step_output('step2_ssl', results if isinstance(results, dict) else {})
        self.progress_tracker.complete_step("ssl", results if isinstance(results, dict) else {})
        return results if isinstance(results, dict) else {}

class XAIStep(StepInterface):
    def execute(self) -> Dict[str, Any]:
        self.progress_tracker.start_step("xai", "Running XAI visualization")
        try:
            xai_module = importlib.import_module(
                "visualization_step_1.xai_visualization.main"
            )
            cli_args = _config_to_cli_args(self.config.get("step_3_xai", {}))
            with _patched_argv(cli_args):
                results = xai_module.main()
        except Exception as exc:  # pragma: no cover
            logger.exception("XAI step failed: %s", exc)
            self.progress_tracker.complete_step("xai", {"error": str(exc)})
            raise

        if results is None:
            results = {}
        self.data_manager.save_step_output('step3_xai', results if isinstance(results, dict) else {})
        self.progress_tracker.complete_step("xai", results if isinstance(results, dict) else {})
        return results if isinstance(results, dict) else {}

class FusionStep(StepInterface):
    def execute(self) -> Dict[str, Any]:
        self.progress_tracker.start_step("fusion", "Running saliency fusion")
        try:
            fusion_module = importlib.import_module(
                "visualization_step_2.fusion_visualization.main_final_fusion"
            )
            cli_args = _config_to_cli_args(self.config.get("step_4_fusion", {}))
            with _patched_argv(cli_args):
                results = fusion_module.main()
        except Exception as exc:  # pragma: no cover
            logger.exception("Fusion step failed: %s", exc)
            self.progress_tracker.complete_step("fusion", {"error": str(exc)})
            raise

        if results is None:
            results = {}
        self.data_manager.save_step_output('step4_fusion', results if isinstance(results, dict) else {})
        self.progress_tracker.complete_step("fusion", results if isinstance(results, dict) else {})
        return results if isinstance(results, dict) else {}
