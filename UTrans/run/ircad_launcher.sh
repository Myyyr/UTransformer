#!/bin/bash
#SBATCH -p public
#SBATCH --gpus=0
#SBATCH --output=data.out # output file name
#SBATCH --error=data.err  # error file name

export nnUNet_raw_data_base="/scratch/lthemyr/nnUNetData/nnUNet_raw"
export nnUNet_preprocessed="/scratch/lthemyr/nnUNetData/nnUNet_preprocessed"
export RESULTS_FOLDER="/scratch/lthemyr/nnUNetData/nnUNet_trained_models"

source /opt/server-env.sh
source /home/lthemyr/cotr/bin/activate



# PROCESS TASK001
srun nnUNet_convert_decathlon_task -i /scratch/lthemyr/Task01_BrainTumour -p 8
# srun nnUNet_plan_and_preprocess -t 082 --verify_dataset_integrity



# cd /home/lthemyr/UTransformer/UTrans/run
# srun python brats_run_all_nnformergt1.py