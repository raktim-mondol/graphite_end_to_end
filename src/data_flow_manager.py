import json
from pathlib import Path
from typing import Dict
import torch

class DataFlowManager:
    """Manage outputs between pipeline steps."""

    def __init__(self, config: Dict):
        self.config = config
        self.outputs = {}

    def save_step_output(self, step_name: str, output_data: Dict):
        """Save output from a pipeline step."""
        output_root = Path(self.config['paths']['output_root']) / step_name
        output_root.mkdir(parents=True, exist_ok=True)

        if 'model' in output_data:
            model = output_data['model']
            torch.save(model, output_root / 'model.pth')

        if 'metrics' in output_data:
            with open(output_root / 'metrics.json', 'w') as f:
                json.dump(output_data['metrics'], f, indent=2)

        self.outputs[step_name] = output_data

    def load_step_output(self, step_name: str) -> Dict:
        """Load output from a previous step."""
        return self.outputs.get(step_name, {})
