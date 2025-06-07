# Reproducibility

To reproduce the published results, follow the guidelines in [REPRODUCIBILITY.md](../REPRODUCIBILITY.md). Key points include:

1. **Environment Setup** – Use the exact package versions listed and set random seeds.
2. **Data Preparation** – Verify the dataset structure and integrity.
3. **Step-by-Step Execution** – Run each pipeline stage in order or use `main_pipeline.py` to automate the process.
4. **Validation** – Compare metrics with the reference results provided in `reference_outputs/`.

A quick reproduction option is available via:
```bash
./quickstart.sh
```
which launches the main pipeline and stores logs under `outputs/`.
