import run_training as run

g = '0'
run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_nnFormerGT1EXTV7', task='017', fold=0, outpath='NNFORMEREXTGT1V7x17', val=False, npz=True, vt_map=(3,5,5,3))
run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_nnFormerGT1EXTV7', task='017', fold=0, outpath='NNFORMEREXTGT1V7x17', val=True,  npz=True, vt_map=(3,5,5,3))
