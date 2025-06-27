# GRAPHITE End-to-End Automation Wiki

A technical handbook for running, customising and debugging the **GRAPHITE** automated pipeline (`main_pipeline.py`).  
Everything below assumes you have cloned the repository and installed dependencies (see `README.md` / `SETUP.md`).

---

## 1 | Introduction to E2E Automation

* **Single entry-point** – execute the full MIL → SSL → XAI → Fusion workflow with one command.  
* **YAML-driven** – no code edits; all parameters live in `config/pipeline_config.yaml`.  
* **Stateless steps** – each stage writes atomic outputs to `outputs/<step>/` and may be re-run independently.  
* **Provenance built-in** – every run produces  
  * `pipeline_progress.json` – real-time status & timing  
  * `pipeline_report.yaml`  – aggregated step metrics

```
Raw data ─► Step-1 MIL ─► Step-2 SSL ─► Step-3 XAI ─► Step-4 Fusion ─► Reports
```

---

## 2 | Pipeline Controller Architecture

| Layer | File | Responsibilities |
|-------|------|------------------|
| CLI front-end | `main_pipeline.py` | parse `--config / --steps`, load YAML, initialise utilities |
| Integration layer | `integration_interfaces.py` | four `StepInterface` subclasses (`MILStep`, `SSLStep`, `XAIStep`, `FusionStep`) |
| Utility services | `src/data_flow_manager.py`, `src/model_manager.py`, `src/progress_tracker.py` | persist artefacts, manage checkpoints, track timing |
| Domain modules | `training_step_*`, `visualization_step_*` | untouched research code invoked via integration layer |

Design highlights  
* **Dynamic importing** – sub-modules loaded only when executed (low memory footprint).  
* **`sys.argv` patching** – wraps existing CLI scripts transparently.  
* **Fail-fast / resume-later** – exception halts pipeline, but previous results persist.

---

## 3 | Configuration Management

Everything is controlled by **`config/pipeline_config.yaml`**.

```yaml
paths:
  data_root: dataset/
  output_root: outputs/
  models_root: models/

step_1_mil:
  epochs: 30
  batch_size: 8
  learning_rate: 5e-4

step_2_ssl:
  data_dir: dataset/ssl_images/
  epochs: 50
  lr: 1e-4

step_3_xai:
  method: gradcam
  wsi_folder: dataset/wsi/
  output_folder: vis_xai/

step_4_fusion:
  cam_method: fullgrad
  fusion_method: confidence
  calculate_metrics: true
```

Guidelines  
* **Boolean flags** – appear only when `true`.  
* **Lists** – become variadic CLI args (`metrics_thresholds: [0.3,0.5]`).  
* **`null`** – key is ignored.  
* Relative paths resolve against repository root.

---

## 4 | Step-by-Step Execution Examples

### Full pipeline
```bash
python main_pipeline.py --config config/pipeline_config.yaml
```

### Specific steps only
```bash
# Re-run MIL and SSL
python main_pipeline.py --config config/pipeline_config.yaml --steps mil ssl
```

### Resume after failure
```bash
python main_pipeline.py --config config/pipeline_config.yaml --steps ssl xai fusion
```

Results structure
```
outputs/
├─ step1_mil/
├─ step2_ssl/
├─ step3_xai/
├─ step4_fusion/
├─ pipeline_progress.json
└─ pipeline_report.yaml
```

---

## 5 | Advanced Usage Patterns

| Scenario | Command / Setting | Notes |
|----------|-------------------|-------|
| Hyper-parameter sweep | loop over multiple YAMLs | outputs timestamp-namespaced |
| CPU-only demo | set `device: cpu` in `step_3_xai`, reduce batch sizes | no GPU required |
| Distributed SSL | shard `step_2_ssl.data_dir`, run controller per node with `--steps ssl` | merge checkpoints later |
| CI smoke test | `python main_pipeline.py --steps mil` with mocked torch | fast validation |
| Dry-run config check | `python main_pipeline.py --steps` (empty) | validates YAML schema only |

---

## 6 | Monitoring & Debugging

### Real-time progress
```bash
tail -f outputs/pipeline_progress.json
```
Fields: `start`, `end`, `duration`, `error`, `results`.

### Logs
* MIL – `training_step_1/mil_training.log`  
* SSL – `training_step_2/self_supervised_training/training.log`  
* XAI – `visualization_step_1/xai_visualization/xai.log`

Set `LOGLEVEL=DEBUG` for verbose output.

### Visual checkpoints
* MIL curves – `outputs/step1_mil/training_history.png`  
* SSL loss – `outputs/step2_ssl/loss.png`  
* XAI heatmaps – `outputs/step3_xai/heatmaps/`

---

## 7 | Integration Layer Technical Details

```python
@contextmanager
def _patched_argv(args):
    orig = sys.argv[:]
    sys.argv = ["pipeline_integration"] + args
    try:
        yield
    finally:
        sys.argv = orig
```

* **`_config_to_cli_args`** – dict → flat CLI list (handles bool / list / None).  
* **Model hand-off** – `ModelManager.save_model()` stores `state_dict` + git commit hash.  
* **DataFlowManager** persists any key named `model` or `metrics` automatically.

---

## 8 | Performance Optimization

| Lever | Effect | Location |
|-------|--------|----------|
| `step_1_mil.batch_size` | GPU memory | YAML |
| `step_2_ssl.num_workers` | CPU data loading | YAML |
| Mixed precision (`--amp`) | 30-40 % speed-up | sub-module CLI |
| Gradient accumulation | simulate large batch | add `grad_acc` in YAML |
| Multi-GPU | launch `torchrun main_pipeline.py ...` | controller is GPU-agnostic |

---

## 9 | Best Practices & Recommended Workflows

1. **Prototype small** (`epochs: 1`), verify outputs, then scale.  
2. **Version YAMLs** – they are immutable experiment manifests.  
3. **Pin seeds** (`REPRODUCIBILITY.md`) for fair comparison.  
4. **Archive env** – `pip freeze > env.txt` alongside results.  
5. **Separate raw vs processed data** (`data_root/raw` vs `.../processed`).

---

## 10 | E2E-Specific Troubleshooting

| Symptom | Likely Cause | Remedy |
|---------|--------------|--------|
| `ModuleNotFoundError: torch` in tests | running on minimal CI image | install torch or rely on mocked tests |
| Step hangs at 0 % GPU | mismatched CUDA / PyTorch build | reinstall matching wheel |
| `pipeline_progress.json` lacks `end` | crash inside step | inspect sub-module log, fix, restart remaining steps |
| Fusion metrics NaN | missing masks or wrong thresholds | verify `mask_dir`, adjust `metrics_thresholds` |
| CUDA OOM during SSL | batch size too high | lower `batch_size`, enable mixed precision |

---

### Need more help?
* Review **`END_TO_END_INTEGRATION_FIX.md`** for in-depth notes.  
* Open a GitHub issue with your `pipeline_progress.json` and system specs.  

Happy automating 🚀
