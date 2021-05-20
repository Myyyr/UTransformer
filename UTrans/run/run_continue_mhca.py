import UTrans.run.nnunet_run_training as run
import os
BASE_DIR="/local/DEEPLEARNING/nnUNetData/nnUNet_trained_models/nnUNet/2d_nnUNetPlansv2.1/Task062_NIHPancreas/"
MODEL_PATH = "fold_1/model_final_checkpoint.model"


mod = "MHCA_v2_IN_LeakyReLU/"
for i in range(1,7):
	pth = os.path.join(BASE_DIR,mod,MODEL_PATH)
	print(pth)
	# exit(0)
	run.main(gpu='0', network='2d', network_trainer='nnUNetTrainerV2_utrans_mhca', task='062', fold=1, outpath='MHCA_v2_c'+str(i), val=False, npz=True, c=False, ep=50, lr=1e-4, pretrained_weights=pth)
	run.main(gpu='0', network='2d', network_trainer='nnUNetTrainerV2_utrans_mhca', task='062', fold=1, outpath='MHCA_v2_c'+str(i), val=True,  npz=True, c=False, ep=50, lr=1e-4 )
	mod = "MHSA_v2_c"+str(i)+"_IN_LeakyReLU/"