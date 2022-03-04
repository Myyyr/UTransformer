import run_training as run
# gpuserver 6

g='0'
run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_nnFormer', task='017', fold=1, outpath='NNFORMER_sup', val=False, npz=True)
run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_nnFormer', task='017', fold=1, outpath='NNFORMER_sup', val=True,  npz=True)
