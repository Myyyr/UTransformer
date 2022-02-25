import run_training as run

# g = '5'
g = '2'
# run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_CoTR_agno', task='017', fold=0, outpath='COTR_agno', val=False, npz=True, na=True)
# run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_CoTR_agno', task='017', fold=0, outpath='COTR_agno', val=True, npz=True, na=True)
# run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_CoTR_agno', task='017', fold=1, outpath='COTR_agno', val=False, npz=True, na=True)
# run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_CoTR_agno', task='017', fold=1, outpath='COTR_agno', val=True, npz=True, na=True)
# run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_CoTR_agno', task='017', fold=2, outpath='COTR_agno', val=False, npz=True, na=True)
# run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_CoTR_agno', task='017', fold=2, outpath='COTR_agno', val=True, npz=True, na=True)
# run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_CoTR_agno', task='017', fold=3, outpath='COTR_agno', val=False, npz=True, na=True)
# run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_CoTR_agno', task='017', fold=3, outpath='COTR_agno', val=True, npz=True, na=True)


run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_CoTR_agno', task='017', fold=1, outpath='nnffCOTR_agno', val=False, npz=True, na=True)
run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_CoTR_agno', task='017', fold=1, outpath='nnffCOTR_agno', val=True, npz=True, na=True)
run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_CoTR_agno', task='017', fold=2, outpath='nnffCOTR_agno', val=False, npz=True, na=True)
run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_CoTR_agno', task='017', fold=2, outpath='nnffCOTR_agno', val=True, npz=True, na=True)
run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_CoTR_agno', task='017', fold=3, outpath='nnffCOTR_agno', val=False, npz=True, na=True)
run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_CoTR_agno', task='017', fold=3, outpath='nnffCOTR_agno', val=True, npz=True, na=True)
