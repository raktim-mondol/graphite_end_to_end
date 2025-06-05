#!/usr/bin/env python3
"""GRAPHITE end-to-end pipeline controller."""
import argparse
import yaml
from pathlib import Path
from typing import Dict
from src.data_flow_manager import DataFlowManager
from src.model_manager import ModelManager
from src.progress_tracker import ProgressTracker
from integration_interfaces import MILStep, SSLStep, XAIStep, FusionStep


def load_config(path: str) -> Dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def generate_pipeline_report(results: Dict, output_dir: str):
    report_path = Path(output_dir) / 'pipeline_report.yaml'
    with open(report_path, 'w') as f:
        yaml.dump(results, f)


def main():
    parser = argparse.ArgumentParser(description="GRAPHITE End-to-End Pipeline")
    parser.add_argument('--config', required=True, help='Path to configuration file')
    parser.add_argument('--steps', nargs='+', choices=['mil', 'ssl', 'xai', 'fusion'],
                        help='Specific steps to run')
    args = parser.parse_args()

    PROJECT_ROOT = Path(__file__).resolve().parent
    config = load_config(args.config)

    models_root_str = config['paths']['models_root']
    models_root = Path(models_root_str)
    if not models_root.is_absolute():
        models_root = PROJECT_ROOT / models_root_str

    output_root_str = config['paths']['output_root']
    output_root = Path(output_root_str)
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root_str

    data_manager = DataFlowManager(config)
    # Ensure model_manager and progress_tracker receive string paths
    model_manager = ModelManager(str(models_root.resolve()))
    progress_tracker = ProgressTracker(str(output_root.resolve()))

    steps = {
        'mil': MILStep(config, data_manager, model_manager, progress_tracker),
        'ssl': SSLStep(config, data_manager, model_manager, progress_tracker),
        'xai': XAIStep(config, data_manager, model_manager, progress_tracker),
        'fusion': FusionStep(config, data_manager, model_manager, progress_tracker)
    }

    steps_to_run = args.steps or ['mil', 'ssl', 'xai', 'fusion']
    results = {}
    for step in steps_to_run:
        print(f"\n🔄 Executing {step.upper()} step...")
        results[step] = steps[step].execute()

    generate_pipeline_report(results, str(output_root))
    print("\n✅ Pipeline completed successfully!")


if __name__ == '__main__':
    main()
