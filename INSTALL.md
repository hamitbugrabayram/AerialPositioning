# 📦 Installation Guide

This guide provides step-by-step instructions for setting up the Satellite Visual Localization benchmark environment.

---

## 📋 Table of Contents

- [System Requirements](#system-requirements)
- [Installation Steps](#installation-steps)
- [Troubleshooting](#troubleshooting)
- [Verification](#verification)

---

## System Requirements

| Component | Requirement |
|-----------|-------------|
| **Operating System** | Linux (Ubuntu 20.04+), Windows 10+, macOS |
| **Python** | 3.9 or higher |
| **GPU** | CUDA-capable GPU (recommended) |
| **CUDA** | 11.7+ (for GPU acceleration) |
| **RAM** | 16GB minimum, 32GB recommended |
| **Storage** | 10GB+ for models and data |

---

## Installation Steps

### Step 1: Clone the Repository

Clone the repository with all submodules:

```bash
git clone --recursive https://github.com/ALFONSOBUGRA/SatelliteLocalization.git
cd SatelliteLocalization
```

> ⚠️ **Important**: The `--recursive` flag is required to download matcher submodules.

<details>
<summary>Already cloned without <code>--recursive</code>?</summary>

Run this command to fetch submodules:
```bash
git submodule update --init --recursive
```

</details>

### Step 2: Create Conda Environment

```bash
# Create a new environment with Python 3.9
conda create -n satellite-loc python=3.9 -y

# Activate the environment
conda activate satellite-loc
```

### Step 3: Install Dependencies

```bash
# Install PyTorch with CUDA support (adjust CUDA version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install remaining dependencies
pip install -r requirements.txt
```

### Step 4: Download Model Weights

Download pretrained weights for your desired matchers:

| Matcher | Download Link | Target Location |
|---------|--------------|-----------------|
| **LoFTR** | [outdoor_ds.ckpt](https://github.com/zju3dv/LoFTR#outdoor-ds) | `matchers/LoFTR/weights/` |
| **GIM** | [GIM Weights](https://github.com/xuelunshen/gim#pretrained-models) | `matchers/gim/weights/` |

> 💡 **Note**: LightGlue and SuperGlue weights are downloaded automatically on first use.

---

## Troubleshooting

<details>
<summary><strong>🔴 CUDA/GPU Issues</strong></summary>

If you encounter GPU-related errors:

```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# If CUDA is not available, try reinstalling PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

</details>

<details>
<summary><strong>🔴 Submodule Import Errors</strong></summary>

If matcher imports fail:

```bash
# Ensure submodules are properly initialized
git submodule update --init --recursive

# Verify submodule directories exist
ls matchers/
```

</details>

<details>
<summary><strong>🔴 Memory Errors</strong></summary>

For out-of-memory errors, try:
- Reducing `resize_target` in `config.yaml`
- Using `device: 'cpu'` (slower but uses less GPU memory)
- Processing fewer images at a time

</details>

---

## Verification

Verify your installation by running:

```bash
# Activate environment
conda activate satellite-loc

# Check imports
python -c "
import torch
import cv2
import numpy as np
print('✅ Core dependencies OK')
print(f'   PyTorch: {torch.__version__}')
print(f'   CUDA: {torch.cuda.is_available()}')
print(f'   OpenCV: {cv2.__version__}')
"
```

You should see output confirming all dependencies are properly installed.

---

## Next Steps

After successful installation:

1. **Configure**: Edit `config.yaml` with your settings
2. **Prepare Data**: Add query and map images to `data/` directory
3. **Run**: Execute `python benchmark.py --config config.yaml`

For detailed usage instructions, see the main [README.md](README.md).