# End-to-End Pipeline Guide

This guide explains how to run the full GRAPHITE pipeline from data preprocessing to final saliency map fusion. The controller script `main_pipeline.py` orchestrates each stage using a unified configuration file and several helper utilities.

## Contents
- [1. Overview](#1-overview)
- [2. Configuration](#2-configuration)
- [3. Running the Pipeline](#3-running-the-pipeline)
- [4. Utility Modules](#4-utility-modules)
- [5. Result Files](#5-result-files)
- [6. Troubleshooting](#6-troubleshooting)

## 1. Overview

The GRAPHITE pipeline is composed of four sequential steps:

1. **MIL Classification** – Train an attention-based MIL model for patient-level diagnosis.
2. **Self-Supervised Learning** – Learn hierarchical graph representations with HierGAT.
3. **XAI Visualization** – Generate heatmaps and explanations using trained models.
4. **Saliency Map Fusion** – Combine attention mechanisms for enhanced interpretability.

The `main_pipeline.py` controller runs these steps in order and tracks progress, model outputs and intermediate metrics.

## 2. Configuration

All parameters and paths are stored in `config/pipeline_config.yaml`:

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
  learning_rate: 0.001
  early_stopping_patience: 10

step_2_ssl:
  epochs: 100
  batch_size: 32
  learning_rate: 0.0001

step_3_xai:
  methods: ["gradcam", "attention", "lime"]
  output_format: ["png", "npy"]

step_4_fusion:
  fusion_method: "weighted_average"
  attention_weights: [0.4, 0.6]
```

Edit this file to customize training parameters or change folder locations.

## 3. Running the Pipeline

Use the quickstart script or invoke the controller directly.

```bash
# Recommended: run from the project root
./quickstart.sh            # choose "Complete pipeline" when prompted

# Or run manually
python main_pipeline.py --config config/pipeline_config.yaml
```

Specify particular steps with `--steps`:

```bash
python main_pipeline.py --config config/pipeline_config.yaml --steps mil ssl
```

Progress information and errors are logged to `outputs/pipeline.log`.

## 4. Utility Modules

Several helper classes keep the pipeline modular:

- **DataFlowManager** – Saves and loads intermediate results between steps.
- **ModelManager** – Stores trained model checkpoints for reuse.
- **ProgressTracker** – Records timing information and writes `pipeline_progress.json`.
- **Integration Interfaces** – Lightweight wrappers that call each existing component.

These modules live in the `src/` directory and are imported by `main_pipeline.py`.

## 5. Result Files

After a successful run you will find outputs organized as follows:

```
outputs/
├── step1_mil/
│   ├── model.pth
│   └── metrics.json
├── step2_ssl/
│   ├── model.pth
│   └── metrics.json
├── step3_xai/
│   └── ... visualization artifacts ...
├── step4_fusion/
│   └── ... fusion outputs ...
├── pipeline_progress.json
└── pipeline_report.yaml
```

Check these files to evaluate performance or resume training.

## 6. Troubleshooting

If the pipeline stops unexpectedly:

1. Inspect `outputs/pipeline.log` for error messages.
2. Ensure dataset directories match `config/pipeline_config.yaml`.
3. Verify Python dependencies with `pip install -r requirements.txt`.
4. For GPU issues, confirm `nvidia-smi` is available and CUDA versions are correct.

---

For a more general introduction to GRAPHITE and detailed explanations of each component, see [README.md](README.md) and the individual READMEs in the subdirectories.
