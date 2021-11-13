import UTrans.run.nnunet_run_training as run
import os
BASE_DIR="/etudiants/siscol/t/themyr_l/nnUNetData/nnUNet_trained_models/nnUNet/3d_fullres_nnUNetPlansv2.1/Task017_BCV/"
MODEL_PATH = "fold_1/model_final_checkpoint.model"


mod = "NNFORMERGT10gv2_IN_LeakyReLU/"
for i in range(1,10):
	pth = os.path.join(BASE_DIR,mod,MODEL_PATH)
	print(pth)
	# exit(0)
	run.main(gpu='1', network='3d_fullres', network_trainer='nnUNetTrainerV2_nnFormerGT10gv2', task='017', fold=1, outpath='NNFORMERGT10gv2_c'+str(i), val=False, npz=True, c=False, ep=50, lr=2e-05, pretrained_weights=pth)
	run.main(gpu='1', network='3d_fullres', network_trainer='nnUNetTrainerV2_nnFormerGT10gv2', task='017', fold=1, outpath='NNFORMERGT10gv2_c'+str(i), val=True,  npz=True, c=False, ep=50, lr=2e-05 )
	mod = "NNFORMERGT10gv2_c"+str(i)+"_IN_LeakyReLU/"