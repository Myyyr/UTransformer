import UTrans.run.nnunet_run_training as run

g = '2'
run.main(gpu=g, network='2d', network_trainer='nnUNetTrainerV2_utrans_imhsa_realtr', task='062', fold=1, outpath='iMHSA_v4', val=False, npz=True)
run.main(gpu=g, network='2d', network_trainer='nnUNetTrainerV2_utrans_imhsa_realtr', task='062', fold=1, outpath='iMHSA_v4', val=True, npz=True)
# run.main(gpu='0', network='2d', network_trainer='nnUNetTrainerV2_utrans_mhsa_realtr', task='062', fold=4, outpath='MHSA_v4', val=False, npz=True)
# run.main(gpu='0', network='2d', network_trainer='nnUNetTrainerV2_utrans_mhsa_realtr', task='062', fold=4, outpath='MHSA_v4', val=True, npz=True)
# run.main(gpu='0', network='2d', network_trainer='nnUNetTrainerV2_utrans_mhsa_realtr', task='062', fold=2, outpath='MHSA_v4', val=False, npz=True)
# run.main(gpu='0', network='2d', network_trainer='nnUNetTrainerV2_utrans_mhsa_realtr', task='062', fold=2, outpath='MHSA_v4', val=True, npz=True)
# run.main(gpu='0', network='2d', network_trainer='nnUNetTrainerV2_utrans_mhsa_realtr', task='062', fold=0, outpath='MHSA_v4', val=False, npz=True)
# run.main(gpu='0', network='2d', network_trainer='nnUNetTrainerV2_utrans_mhsa_realtr', task='062', fold=0, outpath='MHSA_v4', val=True, npz=True)
# run.main(gpu='0', network='2d', network_trainer='nnUNetTrainerV2_utrans_mhsa_realtr', task='062', fold=3, outpath='MHSA_v4', val=False, npz=True)
# run.main(gpu='0', network='2d', network_trainer='nnUNetTrainerV2_utrans_mhsa_realtr', task='062', fold=3, outpath='MHSA_v4', val=True, npz=True)
