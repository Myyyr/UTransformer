import UTrans.run.nnunet_run_training as run
import os
import torch
BASE_DIR="/etudiants/siscol/t/themyr_l/nnUNetData/nnUNet_trained_models/nnUNet/3d_fullres_nnUNetPlansv2.1/Task017_BCV/"
MODEL_PATH = "fold_1/model_final_checkpoint.model"


mod = "NNFORMERGT10gv2_c1_IN_LeakyReLU/"
for i in range(2,10):
	pth = os.path.join(BASE_DIR,mod,MODEL_PATH)
	print(pth)
	# exit(0)
	run.main(gpu='0', network='3d_fullres', network_trainer='nnUNetTrainerV2_nnFormerGT10gv2', task='017', fold=1, outpath='NNFORMERGT10gv2_c'+str(i), val=False, npz=True, c=False, ep=50, lr=2e-05, pretrained_weights=pth)
	torch.cuda.empty_cache()
	run.main(gpu='0', network='3d_fullres', network_trainer='nnUNetTrainerV2_nnFormerGT10gv2', task='017', fold=1, outpath='NNFORMERGT10gv2_c'+str(i), val=True,  npz=True, c=False, ep=50, lr=2e-05 )
	mod = "NNFORMERGT10gv2_c"+str(i)+"_IN_LeakyReLU/"
	torch.cuda.empty_cache()