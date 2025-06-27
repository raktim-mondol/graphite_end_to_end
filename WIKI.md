# GRAPHITE End-to-End Automation Wiki

Welcome to the **GRAPHITE** end-to-end (E2E) automation guide.  
This wiki is **laser-focused** on running, extending and debugging the single-command pipeline that chains:

MIL ➡ SSL ➡ XAI ➡ Fusion

Everything here assumes you have cloned the repo and installed dependencies (see `README.md` / `SETUP.md`).

---

## 1 | Introduction to the E2E Automation

The E2E controller (`main_pipeline.py`) turns four independent research prototypes into **one production-ready workflow**:

* **Single entry-point** – execute the whole pipeline with one command.
* **Configuration-driven** – zero code edits; everything resides in `config/pipeline_config.yaml`.
* **Stateless steps** – each stage writes outputs to disk and can be resumed or skipped.
* **Audit & provenance** – every run outputs `pipeline_progress.json` + `pipeline_report.yaml`.

High-level flow:

```
Raw Data ─► MIL Training ─► SSL HierGAT ─► XAI Visuals ─► Fusion ─► Reports
```

---

## 2 | Pipeline Controller Architecture

| Layer | File | Responsibility |
|-------|------|----------------|
| **CLI Frontend** | `main_pipeline.py` | Parse `--config` / `--steps`, load YAML, initialise utilities |
| **Integration Layer** | `integration_interfaces.py` | Four `StepInterface` subclasses (MIL, SSL, XAI, Fusion) |
| **Utility Services** | `src/` | `DataFlowManager`, `ModelManager`, `ProgressTracker` |
| **Sub-modules** | `training_step_*`, `visualization_step_*` | Domain code – untouched by controller |

Key design decisions:

* **Dynamic Importing** – steps are imported only when executed → low memory footprint.
* **`sys.argv` Patching** – wraps CLI scripts without rewriting their code.
* **Atomic Outputs** – each step writes into `outputs/<step>/` and returns a dict recorded in the report.
* **Fail-fast, Resume-later** – exception in a step halts pipeline but previous results persist.

---

## 3 | Configuration Management & Customization

Everything is centralised in **`config/pipeline_config.yaml`**.

Example (abridged):

```yaml
paths:
  data_root: dataset/
  output_root: outputs/
  models_root: models/

step_1_mil:
  epochs: 30
  batch_size: 8
  learning_rate: 0.0005

step_2_ssl:
  data_dir: dataset/training_dataset_step_2/images/
  epochs: 50
  lr: 1e-4

step_3_xai:
  method: gradcam
  wsi_folder: dataset/wsi_images/
  output_folder: visualization_step_1/output/

step_4_fusion:
  cam_method: fullgrad
  fusion_method: confidence
  calculate_metrics: true
```

**Quick tips**

* **Override on the fly** – keep multiple YAMLs (e.g. `pipeline_gpu.yaml`) and pass with `--config`.
* **Boolean flags** – just add `flag_name: true` to enable CLI switches.
* **Path resolution** – relative paths resolve against repo root.

---

## 4 | Step-by-Step Execution Guide

### Run Everything

```bash
python main_pipeline.py --config config/pipeline_config.yaml
```

### Run Sub-set

```bash
# only SSL and Fusion (skip MIL & XAI)
python main_pipeline.py --config config/pipeline_config.yaml --steps ssl fusion
```

### Hot-restart After Failure

```bash
# suppose SSL crashed; fix config then resume
python main_pipeline.py --config config/pipeline_config.yaml --steps ssl xai fusion
```

---

## 5 | Advanced Usage Patterns & Scenarios

| Scenario | Command / Setting | Notes |
|----------|-------------------|-------|
| **Hyper-param sweep** | create multiple YAMLs and loop `main_pipeline.py` | outputs are namespaced by timestamp |
| **CPU-only demo** | set `device: cpu` in `step_3_xai`; reduce `batch_size` | tests run with mocked torch |
| **Distributed training** | point `step_2_ssl.data_dir` to shared storage, launch controller on each node with `--steps ssl` | each node trains its slice, later merge checkpoints |
| **CI pipeline** | call `main_pipeline.py --steps mil` in GH Actions, use `test_integration.py` for fast validation | no GPU needed |

---

## 6 | Monitoring & Debugging the Pipeline

**Real-time progress**

```bash
tail -f outputs/pipeline_progress.json
```

Keys:

* `start`, `end`, `duration`
* `error` – populated on exceptions
* `results` – arbitrary metrics

**Logs**

Each sub-module keeps its own log file (e.g. `mil_training.log`).  
Set `LOGLEVEL=DEBUG` env var for verbose PyTorch logs.

**Visual checkpoints**

* MIL: `outputs/step1_mil/training_history.png`
* SSL: loss curves in `outputs/step2_ssl/`
* XAI: heatmaps under `outputs/step3_xai/heatmaps/`

---

## 7 | Integration Layer Technical Details

```python
# integration_interfaces.py (excerpt)
def _patched_argv(args):
    original = sys.argv[:]
    sys.argv = ["pipeline_integration"] + args
    try:
        yield
    finally:
        sys.argv = original
```

* **`_config_to_cli_args`** – converts YAML dict ➜ flat CLI list, handles bool / list / None.
* **Model hand-off** – `ModelManager.save_model()` stores `state_dict` + metadata; later steps can load.
* **Data serialization** – any key named `model` or `metrics` in the returned dict is auto-persisted.

---

## 8 | Performance Optimization & Scaling

| Lever | Effect | Where to change |
|-------|--------|-----------------|
| `step_1_mil.batch_size` | GPU memory | YAML |
| `step_2_ssl.num_workers` | data loading CPU usage | YAML |
| Mixed precision | ~40% speedup | enable `--amp` in respective sub-module CLI |
| Gradient accumulation | simulate large batch | add `grad_acc: N` under `step_1_mil` |
| Multi-GPU | launch controller with `torchrun` environment | controller itself is GPU-agnostic |

---

## 9 | Common Workflows & Best Practices

1. **Prototype locally with small epochs** (`epochs: 1`) then scale.
2. **Commit YAML files** – they serve as immutable experiment configs.
3. **Pin random seeds** via `REPRODUCIBILITY.md` to compare runs reliably.
4. **Version checkpoints** – `ModelManager` already embeds `git commit` hash in metadata.
5. **Keep raw & processed data separate** (`data_root/raw`, `data_root/processed`).

---

## 10 | Troubleshooting (E2E Specific)

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| `ModuleNotFoundError: torch` during tests | Running unit tests without PyTorch | `pip install torch` **or** rely on mocked tests only |
| Step hangs at 0% GPU | Wrong CUDA/PyTorch build | reinstall matching CUDA wheel |
| `pipeline_progress.json` missing `end` time | Pipeline crashed mid-step | inspect sub-module log, re-launch with same `--steps` |
| Fusion metrics nan | missing masks | check `mask_dir` path in YAML |
| Out-of-memory during SSL | lower `batch_size`, enable mixed precision | |

---

### Need more help?

* Review detailed fix notes in **`END_TO_END_INTEGRATION_FIX.md`**
* Search open issues or open a new one with the failing `pipeline_progress.json`

Happy researching! 🚀
