#!/bin/bash
# Sequential evaluation for all regions

source /home/hamit/miniconda3/etc/profile.d/conda.sh
conda activate satellite-loc
cd ~/myProjects/AerialPositioning

echo "=========================================="
echo "Starting dataset preparation at $(date)"
echo "=========================================="
python runner.py --dataset-prepare all --tile-provider google esri --map-margin 3000 2>&1
echo "Dataset preparation DONE at $(date)"

echo "=========================================="
echo "Starting Region 1 (auto zoom) at $(date)"
echo "=========================================="
python runner.py --dataset-eval 1 --tile-provider google esri 2>&1
echo "Region 1 DONE at $(date)"

echo "=========================================="
echo "Starting Region 2 (auto zoom) at $(date)"
echo "=========================================="
python runner.py --dataset-eval 2 --tile-provider google esri 2>&1
echo "Region 2 DONE at $(date)"

echo "=========================================="
echo "Starting Region 3 (auto zoom) at $(date)"
echo "=========================================="
python runner.py --dataset-eval 3 --tile-provider google esri 2>&1
echo "Region 3 DONE at $(date)"

echo "=========================================="
echo "Starting Region 4 (auto zoom) at $(date)"
echo "=========================================="
python runner.py --dataset-eval 4 --tile-provider google esri 2>&1
echo "Region 4 DONE at $(date)"

echo "=========================================="
echo "Starting Region 5 (auto zoom) at $(date)"
echo "=========================================="
python runner.py --dataset-eval 5 --tile-provider google esri 2>&1
echo "Region 5 DONE at $(date)"

echo "=========================================="
echo "Starting Region 6 (auto zoom) at $(date)"
echo "=========================================="
python runner.py --dataset-eval 6 --tile-provider google esri 2>&1
echo "Region 6 DONE at $(date)"

echo "=========================================="
echo "Starting Region 7 (auto zoom) at $(date)"
echo "=========================================="
python runner.py --dataset-eval 7 --tile-provider google esri 2>&1
echo "Region 7 DONE at $(date)"

echo "=========================================="
echo "Starting Region 8 (auto zoom) at $(date)"
echo "=========================================="
python runner.py --dataset-eval 8 --tile-provider google esri 2>&1
echo "Region 8 DONE at $(date)"

echo "=========================================="
echo "Starting Region 9 (auto zoom) at $(date)"
echo "=========================================="
python runner.py --dataset-eval 9 --tile-provider google esri 2>&1
echo "Region 9 DONE at $(date)"

echo "=========================================="
echo "Starting Region 10 (auto zoom) at $(date)"
echo "=========================================="
python runner.py --dataset-eval 10 --tile-provider google esri 2>&1
echo "Region 10 DONE at $(date)"

echo "=========================================="
echo "Starting Region 11 (auto zoom) at $(date)"
echo "=========================================="
python runner.py --dataset-eval 11 --tile-provider google esri 2>&1
echo "Region 11 DONE at $(date)"

echo "=========================================="
echo "ALL REGIONS COMPLETE at $(date)"
echo "=========================================="
