import run_training as run

g = '1'



run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_nnFormerGT1EXTV6VT10', task='017', fold=0, outpath='FINEVT10', val=False, npz=True, dbg=True)
run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_nnFormerGT1EXTV6VT10', task='017', fold=0, outpath='FINEVT10', val=True,  npz=True)