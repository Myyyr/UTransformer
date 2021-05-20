import UTrans.run.nnunet_run_training as run


run.main(gpu='2', network='2d', network_trainer='nnUNetTrainerV2_utrans_mhca', task='062', fold=4, outpath='MHCA_v2_c1', val=False, npz=True, c=True, ep=200+50*1, lr=3e-4)
run.main(gpu='2', network='2d', network_trainer='nnUNetTrainerV2_utrans_mhca', task='062', fold=4, outpath='MHCA_v2_c1', val=True,  npz=True, c=True, ep=200+50*1, lr=3e-4)
run.main(gpu='2', network='2d', network_trainer='nnUNetTrainerV2_utrans_mhca', task='062', fold=4, outpath='MHCA_v2_c2', val=False, npz=True, c=True, ep=200+50*2, lr=3e-4)
run.main(gpu='2', network='2d', network_trainer='nnUNetTrainerV2_utrans_mhca', task='062', fold=4, outpath='MHCA_v2_c2', val=True,  npz=True, c=True, ep=200+50*2, lr=3e-4)