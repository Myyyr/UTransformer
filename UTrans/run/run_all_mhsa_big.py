import UTrans.run.nnunet_run_training as run


run.main(gpu='0', network='2d', network_trainer='nnUNetTrainerV2_utrans_mhsa_big', task='062', fold=1, outpath='MHSA_v3', val=False, npz=True)
run.main(gpu='0', network='2d', network_trainer='nnUNetTrainerV2_utrans_mhsa_big', task='062', fold=1, outpath='MHSA_v3', val=True, npz=True)

