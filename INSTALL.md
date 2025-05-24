# Installation Guide

This guide details how to set up the Conda environment required to run the Visual Localization Benchmark.

## 1. Clone the Repository

Open your terminal or command prompt and clone the repository **recursively** to ensure the submodules (LightGlue, SuperGlue, GIM) are also downloaded:

```bash
git clone --recursive https://github.com/ALFONSOBUGRA/SatelliteLocalization.git
cd SatelliteLocalization
```

If you already cloned without --recursive: Navigate into the cloned directory and run:
```bash
git submodule update --init --recursive
```

## 3. Create Conda Environment

Create a conda environment with Python 3.9.

```bash
conda env create -n visloc python=3.9 -y
conda activate visloc
```

After then install all requirements with using "requirements.txt".

```bash
pip install -r requirements.txt
```

Activate Environment: Before running the benchmark script, you must activate the newly created environment:
```bash
conda activate visloc
```