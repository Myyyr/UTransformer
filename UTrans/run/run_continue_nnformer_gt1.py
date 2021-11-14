import UTrans.run.nnunet_run_training as run
import os
import torch
BASE_DIR="/local/DEEPLEARNING/nnUNetData/nnUNetData/nnUNet_trained_models/nnUNet/3d_fullres_nnUNetPlansv2.1/Task017_BCV/"
MODEL_PATH = "fold_1/model_final_checkpoint.model"


mod = "NNFORMERGT1gv2_IN_LeakyReLU/"
for i in range(0,10):
	pth = os.path.join(BASE_DIR,mod,MODEL_PATH)
	print(pth)
	# exit(0)
	run.main(gpu='2', network='3d_fullres', network_trainer='nnUNetTrainerV2_nnFormerGT1gv2', task='017', fold=1, outpath='NNFORMERGT1gv2_c'+str(i), val=False, npz=True, c=False, ep=50, lr=2e-05, pretrained_weights=pth)
	torch.cuda.empty_cache()
	run.main(gpu='2', network='3d_fullres', network_trainer='nnUNetTrainerV2_nnFormerGT1gv2', task='017', fold=1, outpath='NNFORMERGT1gv2_c'+str(i), val=True,  npz=True, c=False, ep=50, lr=2e-05 )
	mod = "NNFORMERGT1gv2_c"+str(i)+"_IN_LeakyReLU/"
	torch.cuda.empty_cache()