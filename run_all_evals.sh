#!/bin/bash
# Sequential evaluation for regions 1, 2, 3, 5, 11
# Run with: nohup bash run_all_evals.sh > logs/eval_all.log 2>&1 &

source /home/bt/anaconda3/etc/profile.d/conda.sh
conda activate aerial-pos
cd /home/bt/GSS/HAMIT/AerialPositioning
export CUDA_VISIBLE_DEVICES=1

echo "=========================================="
echo "Starting Region 1 (auto zoom) at $(date)"
echo "=========================================="
python runner.py --dataset-eval 1 --tile-provider google 2>&1
echo "Region 1 DONE at $(date)"

echo "=========================================="
echo "Starting Region 2 (auto zoom) at $(date)"
echo "=========================================="
python runner.py --dataset-eval 2 --tile-provider google 2>&1
echo "Region 2 DONE at $(date)"

echo "=========================================="
echo "Starting Region 3 (auto zoom) at $(date)"
echo "=========================================="
python runner.py --dataset-eval 3 --tile-provider google 2>&1
echo "Region 3 DONE at $(date)"

echo "=========================================="
echo "Starting Region 5 (auto zoom) at $(date)"
echo "=========================================="
python runner.py --dataset-eval 5 --tile-provider google 2>&1
echo "Region 5 DONE at $(date)"

echo "=========================================="
echo "Starting Region 11 (auto zoom) at $(date)"
echo "=========================================="
python runner.py --dataset-eval 11 --tile-provider google 2>&1
echo "Region 11 DONE at $(date)"

echo "=========================================="
echo "ALL REGIONS COMPLETE at $(date)"
echo "=========================================="
