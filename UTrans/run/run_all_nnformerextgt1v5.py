import run_training as run

g = '0'
run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_nnFormerGT1EXTV5', task='017', fold=1, outpath='NNFORMEREXTGT1V5', val=False, npz=True)
run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_nnFormerGT1EXTV5', task='017', fold=1, outpath='NNFORMEREXTGT1V5', val=True,  npz=True)
