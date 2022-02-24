import run_training as run

g = '1'
run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_nnFormerGT10EXTV6', task='017', fold=4, outpath='NNFORMEREXTGT10V6', val=False, npz=True)
run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_nnFormerGT10EXTV6', task='017', fold=4, outpath='NNFORMEREXTGT10V6', val=True,  npz=True)
