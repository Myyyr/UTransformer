import UTrans.run.nnunet_run_training as run


# run.main(gpu='0', network='2d', network_trainer='nnUNetTrainerV2', task='062', fold=0, outpath='UNET_v2', val=False)
run.main(gpu='0', network='2d', network_trainer='nnUNetTrainerV2', task='062', fold=0, outpath='UNET_v2', val=True, npz=True)
run.main(gpu='0', network='2d', network_trainer='nnUNetTrainerV2', task='062', fold=1, outpath='UNET_v2', val=False, npz=True)
run.main(gpu='0', network='2d', network_trainer='nnUNetTrainerV2', task='062', fold=1, outpath='UNET_v2', val=True, npz=True)
run.main(gpu='0', network='2d', network_trainer='nnUNetTrainerV2', task='062', fold=2, outpath='UNET_v2', val=False, npz=True)
run.main(gpu='0', network='2d', network_trainer='nnUNetTrainerV2', task='062', fold=2, outpath='UNET_v2', val=True, npz=True)
run.main(gpu='0', network='2d', network_trainer='nnUNetTrainerV2', task='062', fold=3, outpath='UNET_v2', val=False, npz=True)
run.main(gpu='0', network='2d', network_trainer='nnUNetTrainerV2', task='062', fold=3, outpath='UNET_v2', val=True, npz=True)
run.main(gpu='0', network='2d', network_trainer='nnUNetTrainerV2', task='062', fold=4, outpath='UNET_v2', val=False, npz=True)
run.main(gpu='0', network='2d', network_trainer='nnUNetTrainerV2', task='062', fold=4, outpath='UNET_v2', val=True, npz=True)
