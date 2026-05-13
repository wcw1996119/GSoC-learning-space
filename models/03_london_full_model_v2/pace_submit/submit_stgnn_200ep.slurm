#!/bin/bash
#SBATCH --job-name=v2_stgnn_200ep
#SBATCH --partition=gpu                  # adjust to your PACE partition (e.g. gpu_v100, gpu_a100)
#SBATCH --gres=gpu:1                      # 1 GPU is plenty (16 GB sufficient)
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00                   # 1 hour wall time (200 epochs ≈ 5-10 min on V100)
#SBATCH --array=42,123,2024               # 3 seeds run as a job array
#SBATCH --output=logs/stgnn_seed%a_%j.out
#SBATCH --error=logs/stgnn_seed%a_%j.err

# Load modules — adjust to your PACE environment
module load python/3.11
module load cuda/12.1
# (or: source activate mesa-gnn  if using conda)

# Set PyTorch to use GPU
export CUDA_VISIBLE_DEVICES=0

# Working directory: assume you've rsync'd the v2 model folder to PACE
cd $SLURM_SUBMIT_DIR

SEED=$SLURM_ARRAY_TASK_ID
echo "Starting STGNN training: seed=$SEED, 200 epochs, hidden=128"

mkdir -p logs ckpt

python train.py \
    --epochs 200 \
    --hidden 128 \
    --gru_hidden 64 \
    --heads 4 \
    --K 10 \
    --lr 1e-3 \
    --beta 0.07 \
    --seed $SEED \
    > logs/stgnn_seed${SEED}_train.log

# Rename the saved checkpoint with seed suffix to avoid collision across array jobs
mv best_model.pt ckpt/best_model_seed${SEED}.pt

echo "Done seed=$SEED"
