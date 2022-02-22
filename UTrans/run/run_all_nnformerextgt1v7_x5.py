import run_training as run

g = '0'
run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_nnFormerGT1EXTV7', task='017', fold=0, outpath='NNFORMEREXTGT1V7x5', val=False, npz=True, vt_map=(3,5,5,1), dbg=True)
# run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_nnFormerGT1EXTV7', task='017', fold=0, outpath='NNFORMEREXTGT1V7x5', val=True,  npz=True, vt_map=(3,5,5,1))
# run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_nnFormerGT1EXTV7', task='017', fold=1, outpath='NNFORMEREXTGT1V7x5', val=False, npz=True, vt_map=(3,5,5,1))
# run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_nnFormerGT1EXTV7', task='017', fold=1, outpath='NNFORMEREXTGT1V7x5', val=True,  npz=True, vt_map=(3,5,5,1))
# run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_nnFormerGT1EXTV7', task='017', fold=2, outpath='NNFORMEREXTGT1V7x5', val=False, npz=True, vt_map=(3,5,5,1))
# run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_nnFormerGT1EXTV7', task='017', fold=2, outpath='NNFORMEREXTGT1V7x5', val=True,  npz=True, vt_map=(3,5,5,1))
# run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_nnFormerGT1EXTV7', task='017', fold=3, outpath='NNFORMEREXTGT1V7x5', val=False, npz=True, vt_map=(3,5,5,1))
# run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_nnFormerGT1EXTV7', task='017', fold=3, outpath='NNFORMEREXTGT1V7x5', val=True,  npz=True, vt_map=(3,5,5,1))
