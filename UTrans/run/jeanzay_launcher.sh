#!/bin/bash
#SBATCH --job-name=city1024     # job name
#SBATCH --ntasks=1                  # number of MP tasks
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --gres=gpu:1                 # number of GPUs per node
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=80:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --qos=qos_gpu-t4
#SBATCH --output=logs/city1024.out # output file name # add %j to id the job
#SBATCH --error=logs/city1024.err  # error file name # add %j to id the job
# # SBATCH -C v100-32g

set -x


cd $WORK/transseg2d
module purge
module load cuda/10.1.2
module load python/3.7.10



srun python -u tools/train.py $CONFIG --resume-from=$RESUME --launcher="slurm" ${@:3} --seed 0 --deterministic --no-validate ${@:3} 
