#!/bin/bash
#SBATCH --job-name=nnff0     # job name
#SBATCH --ntasks=1                  # number of MP tasks
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --gres=gpu:1                 # number of GPUs per node
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=00:05:59              # maximum execution time (HH:MM:SS)
#SBATCH --qos=qos_gpu-dev
#SBATCH --output=logs/nnff0%j.out # output file name
#SBATCH --error=logs/nnff0%j.err  # error file name



set -x


cd "/gpfswork/rech/arf/unm89rb/nnUNet/UTransformer/UTrans/run"
module purge
module load pytorch-gpu/py3/1.10.0

srun export nnUNet_raw_data_base="/gpfsscratch/rech/arf/unm89rb/nnUNetData/nnUNet_raw"
srun export nnUNet_preprocessed="/gpfsscratch/rech/arf/unm89rb/nnUNetData/nnUNet_preprocessed"
srun export RESULTS_FOLDER="/gpfsscratch/rech/arf/unm89rb/nnUNetData/nnUNet_trained_models"

srun python -u  run_all_nnformer.py
