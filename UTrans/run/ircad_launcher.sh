#!/bin/bash
#SBATCH -p public
#SBATCH --gpus=1
#SBATCH --partition=long
#SBATCH --time=100:00:00 
#SBATCH --output=finebraintumor.out # output file name
#SBATCH --error=finebraintumor.err  # error file name


export nnUNet_raw_data_base="/scratch/lthemyr/nnUNetData/nnUNet_raw"
export nnUNet_preprocessed="/scratch/lthemyr/nnUNetData/nnUNet_preprocessed"
export RESULTS_FOLDER="/scratch/lthemyr/nnUNetData/nnUNet_trained_models"

source /opt/server-env.sh
source /home/lthemyr/cotr/bin/activate



## PROCESS TASK001
# srun nnUNet_convert_decathlon_task -i /scratch/lthemyr/Task01_BrainTumour -p 8
# srun nnUNet_plan_and_preprocess -t 001 --verify_dataset_integrity



## Train nnf brats 
cd /home/lthemyr/UTransformer/UTrans/run
# srun python brats_run_all_nnformergt1.py
srun python task001_run_all_fine.py