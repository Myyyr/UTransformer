import run_training as run
# m2n
g='3'
run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_CoTR_agno', task='017', fold=1, outpath='COTR_sup', val=False, npz=True)
run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_CoTR_agno', task='017', fold=1, outpath='COTR_sup', val=True,  npz=True)
