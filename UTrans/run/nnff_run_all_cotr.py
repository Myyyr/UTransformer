import run_training as run

g = '1'
run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_CoTR', task='017', fold=0, outpath='COTR', val=False, npz=True, na=True)
run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_CoTR', task='017', fold=0, outpath='COTR', val=True, npz=True, na=True)
