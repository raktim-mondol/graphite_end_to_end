# Setup

This page summarizes the essential steps for installing and configuring GRAPHITE. For full instructions refer to [SETUP.md](../SETUP.md).

## Requirements
- Python 3.9+
- PyTorch 2.0 or later with CUDA support (GPU recommended)
- See additional packages listed in `requirements.txt`

## Installation
```bash
# Clone repository
git clone https://github.com/your-org/graphite_end_to_end.git
cd graphite_end_to_end

# Install dependencies
pip install -r requirements.txt
```

## Data Setup
Prepare the dataset following the structure described in the [Data Structure](Data_Structure.md) page. Ensure you have sufficient disk space and proper permissions for medical imaging data.

## Verification
After installation run the quickstart script to validate your environment:
```bash
./quickstart.sh
```

Follow the on-screen menu to perform system diagnostics or launch the full pipeline.
