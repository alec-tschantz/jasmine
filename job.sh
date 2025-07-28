#!/usr/bin/env bash
#SBATCH --job-name=tok-train
#SBATCH --nodes=4                        # total nodes
#SBATCH --ntasks-per-node=8              # one task per GPU
#SBATCH --gres=gpu:a100:8                # reserve 8 A100s per node
#SBATCH --cpus-per-task=4                # CPU cores per task
#SBATCH --time=24:00:00                  # walltime
#SBATCH --output=logs/train-%j.out       # stdout & stderr

# --- Load modules / activate your env ---
module load cuda/12.1
module load cudnn/8.5
source ~/envs/jax/bin/activate

# --- JAX distributed setup ---
# pick the first node as coordinator
export JAX_DIST_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
export JAX_DIST_PORT=12345
export JAX_PROCESS_COUNT=$SLURM_NTASKS
export JAX_PROCESS_INDEX=$SLURM_PROCID

# optional XLA tuning
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8

# --- Launch the training script across all GPUs ---
srun python train_tokenizer.py \
     --data_dir /data/episodes \
     --ckpt_dir /data/checkpoints \
     --log \
     --project jasmine-tokenizer
