from pathlib import Path
from typing import Dict, Any
import torch

class ModelManager:
    """Handle saving and loading of models between pipeline steps."""

    def __init__(self, models_dir: str = "models/"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.loaded_models: Dict[str, Any] = {}

    def save_model(self, model: torch.nn.Module, model_name: str, step: str, metadata: Dict = None) -> Path:
        step_dir = self.models_dir / step
        step_dir.mkdir(parents=True, exist_ok=True)
        model_path = step_dir / f"{model_name}.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_class': model.__class__.__name__,
            'metadata': metadata or {}
        }, model_path)
        return model_path

    def load_model(self, model_class, model_name: str, step: str) -> torch.nn.Module:
        model_path = self.models_dir / step / f"{model_name}.pth"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        checkpoint = torch.load(model_path, map_location='cpu')
        model = model_class()
        model.load_state_dict(checkpoint['model_state_dict'])
        self.loaded_models[f"{step}_{model_name}"] = {
            'model': model,
            'metadata': checkpoint.get('metadata', {})
        }
        return model
