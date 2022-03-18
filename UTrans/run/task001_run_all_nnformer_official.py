import run_training as run

g = '1'
run.main(gpu=g, network='3d_fullres', network_trainer='task001_nnUNetTrainerV2_nnFormer_official', task='001', fold=0, outpath='NNFORMEROff', val=False, npz=True, dbg=True)
run.main(gpu=g, network='3d_fullres', network_trainer='task001_nnUNetTrainerV2_nnFormer_official', task='001', fold=0, outpath='NNFORMEROff', val=True,  npz=True)
