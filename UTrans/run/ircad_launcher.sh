#!/bin/bash
#SBATCH -p public
#SBATCH --gpus=1
#SBATCH --partition=long
#SBATCH --time=100:00:00 
#SBATCH --output=logs/nnff4.out # output file name
#SBATCH --error=logs/nnff4.err  # error file name


export nnUNet_raw_data_base="/scratch/lthemyr/nnUNetData/nnUNet_raw"
export nnUNet_preprocessed="/scratch/lthemyr/nnUNetData/nnUNet_preprocessed"
export RESULTS_FOLDER="/scratch/lthemyr/nnUNetData/nnUNet_trained_models"

source /opt/server-env.sh
source /home/lthemyr/cotr/bin/activate



## PROCESS TASK001
# srun nnUNet_convert_decathlon_task -i /scratch/lthemyr/Task01_BrainTumour -p 8
# srun nnUNet_plan_and_preprocess -t 001 --verify_dataset_integrity

## EVAL
# cd /home/lthemyr/UTransformer/UTrans/inference
# srun python tumour.py /scratch/lthemyr FINEV6_2_IN_LeakyReLU
# srun python tumour.py /scratch/lthemyr NNFORMER_IN_LeakyReLU

## TRAIN
# cd /home/lthemyr/UTransformer/UTrans/run
# srun python task001_run_all_fine.py
# srun python task001_run_all_nnformer.py



## Train nnf brats 
# srun python brats_run_all_nnformergt1.py



## TASK 017 BCV
# srun python ircad_run_all_cotr_agno.py
# srun python ircad_run_all_cotr_agno_bis.py
# srun python ircad_run_all_nnformerextgt1v6.py
srun python ircad_run_all_nnformer.py