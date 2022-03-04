import run_training as run
# m2n
# F4 -> F1 (nnf file)
g='3'
run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_CoTR_agno', task='017', fold=4, outpath='COTR_sup', val=False, npz=True, na=True)
run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_CoTR_agno', task='017', fold=4, outpath='COTR_sup', val=True,  npz=True, na=True)
