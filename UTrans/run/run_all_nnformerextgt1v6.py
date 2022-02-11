import run_training as run

g = '0'
# run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_nnFormerGT1EXTV6', task='017', fold=0, outpath='NNFORMEREXTGT1V6', val=False, npz=True)
# run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_nnFormerGT1EXTV6', task='017', fold=0, outpath='NNFORMEREXTGT1V6', val=True,  npz=True)
run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_nnFormerGT1EXTV6', task='017', fold=0, outpath='NNFORMEREXTGT1V6_deter', val=False, npz=True)
run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_nnFormerGT1EXTV6', task='017', fold=0, outpath='NNFORMEREXTGT1V6_deter', val=True,  npz=True)
