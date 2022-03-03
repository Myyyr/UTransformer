import run_training as run

g = '0'


run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2', task='017', fold=4, outpath='nnffNNUNET', val=False, npz=True, na=True)
run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2', task='017', fold=4, outpath='nnffNNUNET', val=True, npz=True, na=True)
