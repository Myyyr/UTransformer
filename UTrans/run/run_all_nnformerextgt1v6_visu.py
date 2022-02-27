import run_training as run

g = '0'
run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_nnFormerGT1EXTV6Visu', task='017', fold=1, outpath='NNFORMEREXTGT1V6Visu', 
	val=True,  npz=True, dbg=True, visu=True)
