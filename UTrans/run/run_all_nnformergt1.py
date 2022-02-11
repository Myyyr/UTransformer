import run_training as run

g = '0'
# run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_nnFormerGT1gv2', task='017', fold=0, outpath='NNFORMERGT1af', val=False, npz=True)
run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_nnFormerGT1gv2', task='017', fold=0, outpath='NNFORMERGT1af', val=True,  npz=True)
# run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_nnFormerGT1gv2', task='017', fold=1, outpath='NNFORMERGT1af', val=False, npz=True)
run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_nnFormerGT1gv2', task='017', fold=1, outpath='NNFORMERGT1af', val=True,  npz=True)
run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_nnFormerGT1gv2', task='017', fold=2, outpath='NNFORMERGT1af', val=False, npz=True)
run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_nnFormerGT1gv2', task='017', fold=2, outpath='NNFORMERGT1af', val=True,  npz=True)
run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_nnFormerGT1gv2', task='017', fold=3, outpath='NNFORMERGT1af', val=False, npz=True)
run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_nnFormerGT1gv2', task='017', fold=3, outpath='NNFORMERGT1af', val=True,  npz=True)
