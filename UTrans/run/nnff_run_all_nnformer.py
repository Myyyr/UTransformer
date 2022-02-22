import run_training as run

g = '0'
run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_nnFormer', task='017', fold=0, outpath='NNFORMER', val=False, npz=True)
run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_nnFormer', task='017', fold=0, outpath='NNFORMER', val=True,  npz=True)
