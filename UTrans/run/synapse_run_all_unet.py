import UTrans.run.nnunet_run_training as run

g='2'
# run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2', task='17', fold=1, outpath='SynpaseUNET_v2', val=False, npz=True)
run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2', task='17', fold=1, outpath='SynpaseUNET_v2', val=True, npz=True)
# run.main(gpu='0', network='2d', network_trainer='nnUNetTrainerV2', task='062', fold=4, outpath='UNET_v2', val=False, npz=True)
# run.main(gpu='0', network='2d', network_trainer='nnUNetTrainerV2', task='062', fold=4, outpath='UNET_v2', val=True, npz=True)
# run.main(gpu='0', network='2d', network_trainer='nnUNetTrainerV2', task='062', fold=2, outpath='UNET_v2', val=False, npz=True)
# run.main(gpu='0', network='2d', network_trainer='nnUNetTrainerV2', task='062', fold=2, outpath='UNET_v2', val=True, npz=True)
# run.main(gpu='0', network='2d', network_trainer='nnUNetTrainerV2', task='062', fold=0, outpath='UNET_v2', val=False)
# run.main(gpu='0', network='2d', network_trainer='nnUNetTrainerV2', task='062', fold=0, outpath='UNET_v2', val=True, npz=True)
# run.main(gpu='0', network='2d', network_trainer='nnUNetTrainerV2', task='062', fold=3, outpath='UNET_v2', val=False, npz=True)
# run.main(gpu='0', network='2d', network_trainer='nnUNetTrainerV2', task='062', fold=3, outpath='UNET_v2', val=True, npz=True)
