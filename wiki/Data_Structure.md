# Data Structure

GRAPHITE expects data to be organised as described in [DATA_STRUCTURE.md](../DATA_STRUCTURE.md). Below is a brief overview.

```
dataset/
├── training_dataset_step_1/
│   └── tma_core/
│       └── <patient_id>/
│           └── patch_001.png
├── training_dataset_step_2/
│   └── core_image/
│       └── <patient_id>/
│           └── image_001.png
└── visualization_dataset/
    └── <patient_id>/
        ├── image.png
        └── mask.png
```

Ensure all image-mask pairs are correctly matched and check for corrupted files. Review the full document for validation checklists and additional advice.
