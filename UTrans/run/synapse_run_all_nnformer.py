import run_training as run

g = '1'
run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_nnFormer', task='17', fold=1, outpath='SynapseNNFORMER', val=False, npz=True)
run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_nnFormer', task='17', fold=1, outpath='SynapseNNFORMER', val=True,  npz=True)
# run.main(gpu='1', network='2d', network_trainer='nnUNetTrainerV2_utrans_mhsa', task='062', fold=4, outpath='MHSA_v2', val=False, npz=True)
# run.main(gpu='1', network='2d', network_trainer='nnUNetTrainerV2_utrans_mhsa', task='062', fold=4, outpath='MHSA_v2', val=True, npz=True)
# run.main(gpu='1', network='2d', network_trainer='nnUNetTrainerV2_utrans_mhsa', task='062', fold=2, outpath='MHSA_v2', val=False, npz=True)
# run.main(gpu='1', network='2d', network_trainer='nnUNetTrainerV2_utrans_mhsa', task='062', fold=2, outpath='MHSA_v2', val=True, npz=True)
# run.main(gpu='1', network='2d', network_trainer='nnUNetTrainerV2_utrans_mhsa', task='062', fold=0, outpath='MHSA_v2', val=False, npz=True)
# run.main(gpu='1', network='2d', network_trainer='nnUNetTrainerV2_utrans_mhsa', task='062', fold=0, outpath='MHSA_v2', val=True, npz=True)
# run.main(gpu='1', network='2d', network_trainer='nnUNetTrainerV2_utrans_mhsa', task='062', fold=3, outpath='MHSA_v2', val=False, npz=True)
# run.main(gpu='1', network='2d', network_trainer='nnUNetTrainerV2_utrans_mhsa', task='062', fold=3, outpath='MHSA_v2', val=True, npz=True)