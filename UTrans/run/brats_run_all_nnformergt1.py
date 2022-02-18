import run_training as run

g = '0'
run.main(gpu=g, network='3d_fullres', network_trainer='brats_nnUNetTrainerV2_nnFormerGT1gv2', task='082', fold=0, outpath='NNFORMERGT1', val=False, npz=True)
run.main(gpu=g, network='3d_fullres', network_trainer='brats_nnUNetTrainerV2_nnFormerGT1gv2', task='082', fold=0, outpath='NNFORMERGT1', val=True,  npz=True)
# run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_nnFormer', task='017', fold=1, outpath='NNFORMERaf', val=False, npz=True)
# run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_nnFormer', task='017', fold=1, outpath='NNFORMERaf', val=True,  npz=True)
# run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_nnFormer', task='017', fold=2, outpath='NNFORMERaf', val=False, npz=True)
# run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_nnFormer', task='017', fold=2, outpath='NNFORMERaf', val=True,  npz=True)
# run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_nnFormer', task='017', fold=3, outpath='NNFORMERaf', val=False, npz=True)
# run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_nnFormer', task='017', fold=3, outpath='NNFORMERaf', val=True,  npz=True)