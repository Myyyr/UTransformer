#!/bin/bash
#SBATCH -p public
#SBATCH --gpus=0
#SBATCH --partition=long
#SBATCH --time=01:00:00 
#SBATCH --output=logs/eval.out # output file name
#SBATCH --error=logs/eval.err  # error file name


export nnUNet_raw_data_base="/scratch/lthemyr/nnUNetData/nnUNet_raw"
export nnUNet_preprocessed="/scratch/lthemyr/nnUNetData/nnUNet_preprocessed"
export RESULTS_FOLDER="/scratch/lthemyr/nnUNetData/nnUNet_trained_models"

source /opt/server-env.sh
source /home/lthemyr/cotr/bin/activate



### PROCESS TASK001
## TRAIN

# srun nnUNet_convert_decathlon_task -i /scratch/lthemyr/Task01_BrainTumour -p 8
# srun nnUNet_plan_and_preprocess -t 001 --verify_dataset_integrity

## EVAL
cd /home/lthemyr/UTransformer/UTrans/inference
srun python tumour.py /scratch/lthemyr

## Train nnf brats 
# cd /home/lthemyr/UTransformer/UTrans/run
# srun python brats_run_all_nnformergt1.py
# srun python task001_run_all_fine.py
# srun python task001_run_all_nnformer.py