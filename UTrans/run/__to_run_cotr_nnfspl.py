import run_training as run
# ircad
# cotr_nnf
g='0'
run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_CoTR_agno', task='017', fold=1, outpath='COTR_nnspl', val=False, npz=True, na=True)
run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_CoTR_agno', task='017', fold=1, outpath='COTR_nnspl', val=True,  npz=True, na=True)

run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_CoTR_agno', task='017', fold=4, outpath='COTR_nnspl', val=False, npz=True, na=True)
run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_CoTR_agno', task='017', fold=4, outpath='COTR_nnspl', val=True,  npz=True, na=True)
