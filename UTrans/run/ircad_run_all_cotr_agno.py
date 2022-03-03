import run_training as run



g = '0'

run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_CoTR_agno', task='017', fold=0, outpath='nnffCOTR_agno', val=False, npz=True, na=True)
run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_CoTR_agno', task='017', fold=0, outpath='nnffCOTR_agno', val=True, npz=True, na=True)
