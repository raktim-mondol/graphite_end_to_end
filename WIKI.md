# GRAPHITE Wiki

## Overview
GRAPHITE (Graph-Based Interpretable Tissue Examination) is a deep learning framework for breast cancer histopathology analysis. It combines multiple instance learning (MIL), hierarchical graph attention networks (HierGAT) and explainable AI (XAI) techniques to provide clinically relevant insights.

## Repository Structure
- `training_step_1/` – MIL classifier implementation
- `training_step_2/` – self-supervised HierGAT training
- `visualization_step_1/` – XAI visualization toolkit
- `visualization_step_2/` – saliency map fusion utilities
- `config/` – pipeline configuration files
- `src/` – orchestration utilities used by `main_pipeline.py`
- `tests/` – unit tests for the pipeline controller

## Getting Started
System requirements include Python 3.9+, PyTorch 2.0.0 and CUDA 11.7. A quick setup is available via `quickstart.sh`. The script checks prerequisites, creates a virtual environment and installs dependencies. For a reproducible demo, follow the commands in `REPRODUCIBILITY.md`.

## Running the Pipeline
`main_pipeline.py` orchestrates the full workflow. The configuration in `config/pipeline_config.yaml` defines paths and parameters. The default pipeline executes four sequential steps:
1. MIL Classification
2. Self-Supervised Learning
3. XAI Visualization
4. Saliency Map Fusion

Example execution:
```bash
python main_pipeline.py --config config/pipeline_config.yaml
```

## Configuration Example
```yaml
pipeline:
  name: "GRAPHITE_Pipeline"
  version: "1.0"
paths:
  data_root: "dataset/"
  output_root: "outputs/"
  models_root: "models/"
step_1_mil:
  epochs: 50
  batch_size: 16
step_2_ssl:
  epochs: 100
  batch_size: 32
step_3_xai:
  methods: ["gradcam", "attention", "lime"]
  output_format: ["png", "npy"]
step_4_fusion:
  fusion_method: "weighted_average"
  attention_weights: [0.4, 0.6]
```

## Data Layout
See `DATA_STRUCTURE.md` for a complete description. Training data is split into three directories:
- `training_dataset_step_1/` – patch-level MIL data
- `training_dataset_step_2/` – full core images for SSL
- `visualization_dataset/` – images for explainability analysis

## Testing
Run the unit tests with:
```bash
python -m unittest tests.test_main_pipeline
```
These tests mock the computational steps and verify that `main_pipeline.py` correctly orchestrates the workflow.

## Reproducibility
To reproduce results quickly, clone the repository and run the quickstart script, then select the options to set up the environment, generate demo data, and run the complete pipeline.

## Further Reading
- [README.md](README.md) – detailed project description
- [END_TO_END_PIPELINE.md](END_TO_END_PIPELINE.md) – step-by-step pipeline guide
- [SETUP.md](SETUP.md) – installation instructions
- [REPRODUCIBILITY.md](REPRODUCIBILITY.md) – guide for reproducing results

## Citation
If you use GRAPHITE in your research, please cite the project as described in `README.md`.

## License
This project is distributed under the MIT License.
