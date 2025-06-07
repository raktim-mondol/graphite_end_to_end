from typing import Dict, Any
from src.data_flow_manager import DataFlowManager
from src.model_manager import ModelManager
from src.progress_tracker import ProgressTracker

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
        from training_step_1.run_training import main as mil_main
        self.progress_tracker.start_step("mil", "Training MIL model")
        results = mil_main()
        if isinstance(results, dict) and 'model' in results:
            self.model_manager.save_model(results['model'], 'mil_model', 'step1',
                                          {'accuracy': results.get('accuracy')})
        self.data_manager.save_step_output('step1_mil', results if isinstance(results, dict) else {})
        self.progress_tracker.complete_step("mil", results if isinstance(results, dict) else {})
        return results if isinstance(results, dict) else {}

class SSLStep(StepInterface):
    def execute(self) -> Dict[str, Any]:
        from training_step_2.self_supervised_training.train import main as ssl_main
        self.progress_tracker.start_step("ssl", "Training HierGAT model")
        results = ssl_main()
        if isinstance(results, dict) and 'model' in results:
            self.model_manager.save_model(results['model'], 'hiergat_model', 'step2')
        self.data_manager.save_step_output('step2_ssl', results if isinstance(results, dict) else {})
        self.progress_tracker.complete_step("ssl", results if isinstance(results, dict) else {})
        return results if isinstance(results, dict) else {}

class XAIStep(StepInterface):
    def execute(self) -> Dict[str, Any]:
        from visualization_step_1.xai_visualization.main import main as xai_main
        self.progress_tracker.start_step("xai", "Running XAI visualization")
        results = xai_main()
        self.data_manager.save_step_output('step3_xai', results if isinstance(results, dict) else {})
        self.progress_tracker.complete_step("xai", results if isinstance(results, dict) else {})
        return results if isinstance(results, dict) else {}

class FusionStep(StepInterface):
    def execute(self) -> Dict[str, Any]:
        from visualization_step_2.fusion_visualization.main_final_fusion import main as fusion_main
        self.progress_tracker.start_step("fusion", "Running saliency fusion")
        results = fusion_main()
        self.data_manager.save_step_output('step4_fusion', results if isinstance(results, dict) else {})
        self.progress_tracker.complete_step("fusion", results if isinstance(results, dict) else {})
        return results if isinstance(results, dict) else {}
