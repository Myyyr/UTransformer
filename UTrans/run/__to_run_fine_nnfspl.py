import run_training as run
# ircad
# fine
g='0'
run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_nnFormerGT1EXTV6', task='017', fold=1, outpath='FINE_nnspl', val=False, npz=True)
run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_nnFormerGT1EXTV6', task='017', fold=1, outpath='FINE_nnspl', val=True,  npz=True)

run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_nnFormerGT1EXTV6', task='017', fold=4, outpath='FINE_nnspl', val=False, npz=True)
run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_nnFormerGT1EXTV6', task='017', fold=4, outpath='FINE_nnspl', val=True,  npz=True)
