import run_training as run

g = '2'
run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_CoTR', task='017', fold=0, outpath='COTRaf', val=False, npz=True, na=True)
run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_CoTR', task='017', fold=0, outpath='COTRaf', val=True, npz=True, na=True)
run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_CoTR', task='017', fold=1, outpath='COTRaf', val=False, npz=True, na=True)
run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_CoTR', task='017', fold=1, outpath='COTRaf', val=True, npz=True, na=True)
run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_CoTR', task='017', fold=2, outpath='COTRaf', val=False, npz=True, na=True)
run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_CoTR', task='017', fold=2, outpath='COTRaf', val=True, npz=True, na=True)
run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_CoTR', task='017', fold=3, outpath='COTRaf', val=False, npz=True, na=True)
run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_CoTR', task='017', fold=3, outpath='COTRaf', val=True, npz=True, na=True)
