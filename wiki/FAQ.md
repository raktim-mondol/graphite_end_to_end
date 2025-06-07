# FAQ

### How do I run only a single pipeline step?
Pass the `--steps` argument to `main_pipeline.py` with the desired stage, e.g. `python main_pipeline.py --config config/pipeline_config.yaml --steps mil`.

### Where are models saved?
Models are stored under the `models/` directory, organised by pipeline step.

### My dataset fails validation. What should I check?
Make sure your directories match the layout shown on the [Data Structure](Data_Structure.md) page and that no files are corrupted.

### Is GPU required?
GPU acceleration is recommended for training but the code can run on CPU with longer runtimes.

### How do I cite GRAPHITE?
See the citation section in the main [README](../README.md#%F0%9F%93%88-citation).
