# import UTrans.run.nnunet_run_training as run
import run_training as run
g = '3'

run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2', task='017', fold=0, outpath='NNUNETaf', val=False, npz=True, na=True)
run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2', task='017', fold=0, outpath='NNUNETaf', val=True, npz=True, na=True)
run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2', task='017', fold=1, outpath='NNUNETaf', val=False, npz=True, na=True)
run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2', task='017', fold=1, outpath='NNUNETaf', val=True, npz=True, na=True)
run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2', task='017', fold=2, outpath='NNUNETaf', val=False, npz=True, na=True)
run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2', task='017', fold=2, outpath='NNUNETaf', val=True, npz=True, na=True)
run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2', task='017', fold=3, outpath='NNUNETaf', val=False, npz=True, na=True)
run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2', task='017', fold=3, outpath='NNUNETaf', val=True, npz=True, na=True)

