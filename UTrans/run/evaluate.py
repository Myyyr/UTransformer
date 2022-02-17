import run_training as run
import telegram_send as ts

g = '5'
ts.send(messages=["Evaluation on m2n gpu"+g+"started!"])
ts.send(messages=["nnUNetTrainerV2_nnFormerGT1EXTV6 TASK017 Fold_0 NNFORMEREXTGT1V6af ..."])
run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_nnFormerGT1EXTV6', task='017', fold=0, outpath='NNFORMEREXTGT1V6af', val=True,  npz=True)
ts.send(messages=["nnUNetTrainerV2_nnFormerGT1EXTV6 TASK017 Fold_0 NNFORMEREXTGT1V6af : END!"])


ts.send(messages=["nnUNetTrainerV2_nnFormerGT1gv2 TASK017 Fold_0 NNFORMERGT1af ..."])
run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_nnFormerGT1gv2', task='017', fold=0, outpath='NNFORMERGT1af', val=True,  npz=True)
ts.send(messages=["nnUNetTrainerV2_nnFormerGT1gv2 TASK017 Fold_0 NNFORMERGT1af : END!"])

ts.send(messages=["nnUNetTrainerV2_nnFormer TASK017 Fold_0 NNFORMERaf ..."])
run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_nnFormer', task='017', fold=0, outpath='NNFORMERaf', val=True,  npz=True)
ts.send(messages=["nnUNetTrainerV2_nnFormer TASK017 Fold_0 NNFORMERaf : END!"])
