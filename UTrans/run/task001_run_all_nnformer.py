import run_training as run

g = '0'
run.main(gpu=g, network='3d_fullres', network_trainer='task001_nnUNetTrainerV2_nnFormer', task='001', fold=0, outpath='NNFORMER', val=False, npz=True)
run.main(gpu=g, network='3d_fullres', network_trainer='task001_nnUNetTrainerV2_nnFormer', task='001', fold=0, outpath='NNFORMER', val=True,  npz=True)
