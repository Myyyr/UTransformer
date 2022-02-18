#!/bin/bash
#SBATCH -p public
#SBATCH --gpus=1
#SBATCH --output=medtest.out # output file name
#SBATCH --error=medtest.err  # error file name

export nnUNet_raw_data_base="/scratch/lthemyr/nnUNetData/nnUNet_raw"
export nnUNet_preprocessed="/scratch/lthemyr/nnUNetData/nnUNet_preprocessed"
export RESULTS_FOLDER="/scratch/lthemyr/nnUNetData/nnUNet_trained_models"

source /opt/server-env.sh
source /home/lthemyr/cotr/bin/activate

cd /home/lthemyr/nnUNet/nnunet/dataset_conversion

# srun python Task082_BraTS_2020.py
# srun nnUNet_plan_and_preprocess -t 082 --verify_dataset_integrity
srun python brats_run_all_nnformergt1.py