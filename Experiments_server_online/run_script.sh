#!/bin/sh
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=512
#SBATCH --mail-type=END

/usr/bin/scontrol show job -d "$SLURM_JOB_ID"

srun python pace_experiments.py $1 $2 $3
