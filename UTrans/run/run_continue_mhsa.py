import UTrans.run.nnunet_run_training as run
import os
BASE_DIR="/local/DEEPLEARNING/nnUNetData/nnUNet_trained_models/nnUNet/2d_nnUNetPlansv2.1/Task062_NIHPancreas/"
MODEL_PATH = "fold_1/model_final_checkpoint.model"


mod = "MHSA_v2_IN_LeakyReLU/"
for i in range(1,7):
	pth = os.path.join(BASE_DIR,mod,MODEL_PATH)
	print(pth)
	exit(0)
	run.main(gpu='1', network='2d', network_trainer='nnUNetTrainerV2_utrans_mhsa', task='062', fold=1, outpath='MHSA_v2_c'+str(i), val=False, npz=True, c=True, ep=200+50*i, lr=3e-4, pretrained_weights=pth)
	run.main(gpu='1', network='2d', network_trainer='nnUNetTrainerV2_utrans_mhsa', task='062', fold=1, outpath='MHSA_v2_c'+str(i), val=True,  npz=True, c=True, ep=200+50*i, lr=3e-4 )
	mod = "MHSA_v2_c"+str(i)+"_IN_LeakyReLU/"
