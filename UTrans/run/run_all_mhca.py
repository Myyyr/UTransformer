import UTrans.run.nnunet_run_training as run


run.main(gpu='2', network='2d', network_trainer='nnUNetTrainerV2_utrans_mhca', task='062', fold=1, outpath='MHCA_v2', val=False, npz=True)
run.main(gpu='2', network='2d', network_trainer='nnUNetTrainerV2_utrans_mhca', task='062', fold=1, outpath='MHCA_v2', val=True, npz=True)
run.main(gpu='2', network='2d', network_trainer='nnUNetTrainerV2_utrans_mhca', task='062', fold=4, outpath='MHCA_v2', val=False, npz=True)
run.main(gpu='2', network='2d', network_trainer='nnUNetTrainerV2_utrans_mhca', task='062', fold=4, outpath='MHCA_v2', val=True, npz=True)
run.main(gpu='2', network='2d', network_trainer='nnUNetTrainerV2_utrans_mhca', task='062', fold=2, outpath='MHCA_v2', val=False, npz=True)
run.main(gpu='2', network='2d', network_trainer='nnUNetTrainerV2_utrans_mhca', task='062', fold=2, outpath='MHCA_v2', val=True, npz=True)
run.main(gpu='2', network='2d', network_trainer='nnUNetTrainerV2_utrans_mhca', task='062', fold=0, outpath='MHCA_v2', val=False, npz=True)
run.main(gpu='2', network='2d', network_trainer='nnUNetTrainerV2_utrans_mhca', task='062', fold=0, outpath='MHCA_v2', val=True, npz=True)
run.main(gpu='2', network='2d', network_trainer='nnUNetTrainerV2_utrans_mhca', task='062', fold=3, outpath='MHCA_v2', val=False, npz=True)
run.main(gpu='2', network='2d', network_trainer='nnUNetTrainerV2_utrans_mhca', task='062', fold=3, outpath='MHCA_v2', val=True, npz=True)
