import run_training as run

g = '0'
run.main(gpu=g, network='3d_fullres', network_trainer='task001_nnUNetTrainerV2_nnFormerGT1EXTV6', task='001', fold=0, outpath='FINEV6_2', val=False, npz=True)
run.main(gpu=g, network='3d_fullres', network_trainer='task001_nnUNetTrainerV2_nnFormerGT1EXTV6', task='001', fold=0, outpath='FINEV6_2', val=True,  npz=True)
