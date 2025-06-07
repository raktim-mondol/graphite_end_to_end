# Pipeline

GRAPHITE uses a four-stage pipeline that can be executed through [`main_pipeline.py`](../main_pipeline.py) or the `quickstart.sh` script.

1. **MIL Classification** – trains an attention-based model using code in [`training_step_1`](../training_step_1). The result is a MIL model checkpoint and evaluation metrics.
2. **Self-Supervised Learning** – learns hierarchical graph representations via the HierGAT model in [`training_step_2`](../training_step_2).
3. **XAI Visualization** – generates attention and saliency maps with scripts in [`visualization_step_1`](../visualization_step_1).
4. **Saliency Map Fusion** – fuses outputs from previous steps for enhanced interpretability, implemented in [`visualization_step_2`](../visualization_step_2).

Run the entire workflow with:
```bash
python main_pipeline.py --config config/pipeline_config.yaml
```

Each step stores outputs under the `outputs/` directory and models under `models/`. Progress is tracked in `outputs/pipeline_progress.json`.
