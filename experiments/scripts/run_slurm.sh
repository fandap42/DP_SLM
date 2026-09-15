#!/bin/bash
#SBATCH --job-name=slm_tokipona
#SBATCH --output=results/metrics/slurm_%j.log
#SBATCH --error=results/metrics/slurm_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00

echo "=========================================================="
echo "Starting SLM Toki Pona Factorial Experiment on SLURM"
echo "Host: $(hostname)"
echo "CUDA / GPU Device: $CUDA_VISIBLE_DEVICES"
echo "Date: $(date)"
echo "=========================================================="

# Load module environment if available on cluster
# module load Python/3.11-GCCcore-12.3.0
# module load CUDA/12.1.1

# Activate virtual environment
source .venv/bin/activate || source venv/bin/activate

# Execute full factorial grid
python experiments/scripts/run_experiments.py --config experiments/configs/factorial_full.json

echo "Job completed at $(date)"
