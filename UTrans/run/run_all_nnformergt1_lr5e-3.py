import run_training as run

g = '0'
run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_nnFormerGT1gv2', task='017', fold=1, outpath='NNFORMERGT15e_3', val=False, npz=True, lr=5e-3)
run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_nnFormerGT1gv2', task='017', fold=1, outpath='NNFORMERGT15e_3', val=True,  npz=True, lr=5e-3)
