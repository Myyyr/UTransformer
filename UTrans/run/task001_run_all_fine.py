import run_training as run

g = '0'
run.main(gpu=g, network='3d_fullres', network_trainer='task001_nnUNetTrainerV2_nnFormerGT1EXTV6.py', task='001', fold=0, outpath='FINEV6', val=False, npz=True, dbg=True)
run.main(gpu=g, network='3d_fullres', network_trainer='task001_nnUNetTrainerV2_nnFormerGT1EXTV6.py', task='001', fold=0, outpath='FINEV6', val=True,  npz=True)
