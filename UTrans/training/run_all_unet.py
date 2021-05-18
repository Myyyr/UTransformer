import Utrans.run.run_training_script as run


run.main(gpu='0', network='2d', network_trainer='nnUNetTrainerV2', task='062', fold=0, outpath='UNET_v2', val=False)
run.main(gpu='0', network='2d', network_trainer='nnUNetTrainerV2', task='062', fold=0, outpath='UNET_v2', val=True)
run.main(gpu='0', network='2d', network_trainer='nnUNetTrainerV2', task='062', fold=1, outpath='UNET_v2', val=False)
run.main(gpu='0', network='2d', network_trainer='nnUNetTrainerV2', task='062', fold=1, outpath='UNET_v2', val=True)
run.main(gpu='0', network='2d', network_trainer='nnUNetTrainerV2', task='062', fold=2, outpath='UNET_v2', val=False)
run.main(gpu='0', network='2d', network_trainer='nnUNetTrainerV2', task='062', fold=2, outpath='UNET_v2', val=True)
run.main(gpu='0', network='2d', network_trainer='nnUNetTrainerV2', task='062', fold=3, outpath='UNET_v2', val=False)
run.main(gpu='0', network='2d', network_trainer='nnUNetTrainerV2', task='062', fold=3, outpath='UNET_v2', val=True)
run.main(gpu='0', network='2d', network_trainer='nnUNetTrainerV2', task='062', fold=4, outpath='UNET_v2', val=False)
run.main(gpu='0', network='2d', network_trainer='nnUNetTrainerV2', task='062', fold=4, outpath='UNET_v2', val=True)
