#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


# from collections import OrderedDict
# from typing import Tuple


# import numpy as np
# import torch
# from nnunet.training.data_augmentation.data_augmentation_moreDA_real import get_moreDA_augmentation
# from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
# from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
# # from nnunet.network_architecture.nnFormer_synapse import nnFormer
# from nnunet.network_architecture.initialization import InitWeights_He
# from nnunet.network_architecture.neural_network import SegmentationNetwork
# from nnunet.training.data_augmentation.default_data_augmentation import default_2D_augmentation_params, \
#     get_patch_size, default_3D_augmentation_params
# from nnunet.training.dataloading.dataset_loading import unpack_dataset
# from UTrans.training.network_training.nnUNetTrainer import nnUNetTrainer
# from nnunet.utilities.nd_softmax import softmax_helper
# from sklearn.model_selection import KFold
# from torch import nn
# from torch.cuda.amp import autocast
# from nnunet.training.learning_rate.poly_lr import poly_lr
# from batchgenerators.utilities.file_and_folder_operations import *

# from UTrans.network_architecture.nnformer import swintransformer

from collections import OrderedDict
from typing import Tuple

import numpy as np
import torch
import shutil
from nnunet.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.training.data_augmentation.default_data_augmentation import default_2D_augmentation_params, \
    get_patch_size, default_3D_augmentation_params
from nnunet.training.dataloading.dataset_loading import unpack_dataset
from UTrans.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.utilities.nd_softmax import softmax_helper
from sklearn.model_selection import KFold
from torch import nn
from torch.cuda.amp import autocast
from nnunet.training.learning_rate.poly_lr import poly_lr
from batchgenerators.utilities.file_and_folder_operations import *

# from UTrans.network_architecture.nnformer import swintransformer
from UTrans.network_architecture.nnformer_official import nnFormer



class task001_nnUNetTrainerV2_nnFormer_official(nnUNetTrainer):
    """
    Info for Fabian: same as internal nnUNetTrainerV2_2
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, vt_map=None):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.max_num_epochs = 1000
        self.initial_lr = 1e-2
        self.deep_supervision_scales = None
        self.ds_loss_weights = None
        self.pin_memory = True
        self.load_pretrain_weight=False
        
        self.load_plans_file()    
        
        if len(self.plans['plans_per_stage'])==2:
            Stage=1
        else:
            Stage=0
            
        self.crop_size=self.plans['plans_per_stage'][Stage]['patch_size']
        # print(self.plans['plans_per_stage'][Stage]['patch_size'])
        # exit(0)
        self.input_channels=self.plans['num_modalities']
        self.num_classes=self.plans['num_classes'] + 1
        self.conv_op=nn.Conv3d
        
        self.embedding_dim=96
        self.depths=[2, 2, 2, 2]
        self.num_heads=[3, 6, 12, 24]
        self.embedding_patch_size=[4,4,4]
        self.window_size=[4,4,8,4]
        
        self.deep_supervision=False
    def initialize(self, training=True, force_load_plans=False):
        """
        - replaced get_default_augmentation with get_moreDA_augmentation
        - enforce to only run this code once
        - loss function wrapper for deep supervision
        :param training:
        :param force_load_plans:
        :return:
        """
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or (self.plans is None):
                self.load_plans_file()
            
            self.process_plans(self.plans)

            self.setup_DA_params()
            if self.deep_supervision:
                ################# Here we wrap the loss for deep supervision ############
                # we need to know the number of outputs of the network
                net_numpool = len(self.net_num_pool_op_kernel_sizes)

                # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
                # this gives higher resolution outputs more weight in the loss
                weights = np.array([1 / (2 ** i) for i in range(net_numpool)])

                # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
                #mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
                #weights[~mask] = 0
                weights = weights / weights.sum()
                print(weights)
                self.ds_loss_weights = weights
                # now wrap the loss
                self.loss = MultipleOutputLoss2(self.loss, self.ds_loss_weights)
                ################# END ###################
            
            self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +"_stage%d" % self.stage)
            seeds_train = np.random.random_integers(0, 99999, self.data_aug_params.get('num_threads'))
            seeds_val = np.random.random_integers(0, 99999, max(self.data_aug_params.get('num_threads') // 2, 1))                         
            if training:
                self.dl_tr, self.dl_val = self.get_basic_generators()
                if self.unpack_data:
                    print("unpacking dataset")
                    unpack_dataset(self.folder_with_preprocessed_data)
                    print("done")
                else:
                    print(
                        "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                        "will wait all winter for your model to finish!")

                self.tr_gen, self.val_gen = get_moreDA_augmentation(
                    self.dl_tr, self.dl_val,
                    self.data_aug_params[
                        'patch_size_for_spatialtransform'],
                    self.data_aug_params,
                    deep_supervision_scales=self.deep_supervision_scales if self.deep_supervision else None,
                    pin_memory=self.pin_memory,
                    use_nondetMultiThreadedAugmenter=False,
                    seeds_train=seeds_train,
                    seeds_val=seeds_val
                )
                self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                                       also_print_to_console=False)
                self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                                       also_print_to_console=False)
            else:
                pass

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()

            assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
        self.was_initialized = True

    def initialize_network(self):
        """
        - momentum 0.99
        - SGD instead of Adam
        - self.lr_scheduler = None because we do poly_lr
        - deep supervision = True
        - i am sure I forgot something here
        Known issue: forgot to set neg_slope=0 in InitWeights_He; should not make a difference though
        :return:
        """
        conv_op = nn.Conv3d
        dropout_op = nn.Dropout3d
        norm_op = nn.InstanceNorm3d
        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
      
        
        self.network=nnFormer(crop_size=self.crop_size,
                                embedding_dim=self.embedding_dim,
                                input_channels=self.input_channels,
                                num_classes=self.num_classes,
                                conv_op=self.conv_op,
                                depths=self.depths,
                                num_heads=self.num_heads,
                                patch_size=self.embedding_patch_size,
                                window_size=self.window_size,
                                deep_supervision=self.deep_supervision)

        if self.load_pretrain_weight:
            checkpoint = torch.load("/home/xychen/jsguo/weight/tumor_pretrain.model", map_location='cpu')
            ck={}
            
            for i in self.network.state_dict():
                if i in checkpoint:
                    print(i)
                    ck.update({i:checkpoint[i]})
                else:
                    ck.update({i:self.network.state_dict()[i]})
            self.network.load_state_dict(ck)
            print('I am using the pre_train weight!!')
        
     
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper
        
    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                         momentum=0.99, nesterov=True)
        self.lr_scheduler = None

    def run_online_evaluation(self, output, target):
        """
        due to deep supervision the return value and the reference are now lists of tensors. We only need the full
        resolution output because this is what we are interested in in the end. The others are ignored
        :param output:
        :param target:
        :return:
        """
        if self.deep_supervision:
            target = target[0]
            output = output[0]
        else:
            target = target
            output = output
        return super().run_online_evaluation(output, target)

    def validate(self, do_mirroring: bool = True, use_sliding_window: bool = True,
                 step_size: float = 0.5, save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True):
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """
        ds = self.network.do_ds
        self.network.do_ds = False
        # ret = super().validate(do_mirroring=do_mirroring, use_sliding_window=use_sliding_window, step_size=step_size,
        #                        save_softmax=save_softmax, use_gaussian=use_gaussian,
        #                        overwrite=overwrite, validation_folder_name=validation_folder_name, debug=debug,
        #                        all_in_gpu=all_in_gpu, segmentation_export_kwargs=segmentation_export_kwargs,
        #                        run_postprocessing_on_folds=run_postprocessing_on_folds)

        ret = super().validate(do_mirroring=do_mirroring, use_sliding_window=use_sliding_window, step_size=step_size,
                               save_softmax=save_softmax, use_gaussian=use_gaussian,
                               overwrite=overwrite, validation_folder_name=validation_folder_name, debug=debug,
                               all_in_gpu=all_in_gpu, segmentation_export_kwargs=segmentation_export_kwargs)
                               # ,run_postprocessing_on_folds=run_postprocessing_on_folds)

        self.network.do_ds = ds
        return ret

    def predict_preprocessed_data_return_seg_and_softmax(self, data: np.ndarray, do_mirroring: bool = True,
                                                         mirror_axes: Tuple[int] = None,
                                                         use_sliding_window: bool = True, step_size: float = 0.5,
                                                         use_gaussian: bool = True, pad_border_mode: str = 'constant',
                                                         pad_kwargs: dict = None, all_in_gpu: bool = False,
                                                         verbose: bool = True, mixed_precision=True) -> Tuple[np.ndarray, np.ndarray]:
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """
        ds = self.network.do_ds
        self.network.do_ds = False
        ret = super().predict_preprocessed_data_return_seg_and_softmax(data,
                                                                       do_mirroring=do_mirroring,
                                                                       mirror_axes=mirror_axes,
                                                                       use_sliding_window=use_sliding_window,
                                                                       step_size=step_size, use_gaussian=use_gaussian,
                                                                       pad_border_mode=pad_border_mode,
                                                                       pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu,
                                                                       verbose=verbose,
                                                                       mixed_precision=mixed_precision)
        self.network.do_ds = ds
        return ret

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        """
        gradient clipping improves training stability
        :param data_generator:
        :param do_backprop:
        :param run_online_evaluation:
        :return:
        """
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)

        self.optimizer.zero_grad()

        if self.fp16:
            with autocast():
                output = self.network(data)
                del data
                
                l = self.loss(output, target)

            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            output = self.network(data)
            del data
            l = self.loss(output, target)

            if do_backprop:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        del target

        return l.detach().cpu().numpy()

    def do_split(self):
        """
        The default split is a 5 fold CV on all available training cases. nnU-Net will create a split (it is seeded,
        so always the same) and save it as splits_final.pkl file in the preprocessed data directory.
        Sometimes you may want to create your own split for various reasons. For this you will need to create your own
        splits_final.pkl file. If this file is present, nnU-Net is going to use it and whatever splits are defined in
        it. You can create as many splits in this file as you want. Note that if you define only 4 splits (fold 0-3)
        and then set fold=4 when training (that would be the fifth split), nnU-Net will print a warning and proceed to
        use a random 80:20 data split.
        :return:
        """
        if self.fold == "all":
            # if fold==all then we use all images for training and validation
            tr_keys = val_keys = list(self.dataset.keys())
        else:
            splits_file = join(self.dataset_directory, "splits_final.pkl")

            # if the split file does not exist we need to create it
            if not isfile(splits_file):
                self.print_to_log_file("Creating new 5-fold cross-validation split...")
                splits = []
                all_keys_sorted = np.sort(list(self.dataset.keys()))
                kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
                for i, (train_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):
                    train_keys = np.array(all_keys_sorted)[train_idx]
                    test_keys = np.array(all_keys_sorted)[test_idx]
                    splits.append(OrderedDict())
                    splits[-1]['train'] = train_keys
                    splits[-1]['val'] = test_keys
                save_pickle(splits, splits_file)

            else:
                self.print_to_log_file("Using splits from existing split file:", splits_file)
                splits = load_pickle(splits_file)
                self.print_to_log_file("The split file contains %d splits." % len(splits))

            self.print_to_log_file("Desired fold for training: %d" % self.fold)
       #      splits[self.fold]['train']=np.array(['BraTS20_Training_001', 'BraTS20_Training_002', 'BraTS20_Training_003', 'BraTS20_Training_004', 'BraTS20_Training_005',
       # 'BraTS20_Training_006', 'BraTS20_Training_007', 'BraTS20_Training_008', 'BraTS20_Training_009', 'BraTS20_Training_010',
       # 'BraTS20_Training_013', 'BraTS20_Training_014', 'BraTS20_Training_015', 'BraTS20_Training_016', 'BraTS20_Training_017',
       # 'BraTS20_Training_019', 'BraTS20_Training_022', 'BraTS20_Training_023', 'BraTS20_Training_024', 'BraTS20_Training_025',
       # 'BraTS20_Training_026', 'BraTS20_Training_027', 'BraTS20_Training_030', 'BraTS20_Training_031', 'BraTS20_Training_033',
       # 'BraTS20_Training_035', 'BraTS20_Training_037', 'BraTS20_Training_038', 'BraTS20_Training_039', 'BraTS20_Training_040',
       # 'BraTS20_Training_042', 'BraTS20_Training_043', 'BraTS20_Training_044', 'BraTS20_Training_045', 'BraTS20_Training_046',
       # 'BraTS20_Training_048', 'BraTS20_Training_050', 'BraTS20_Training_051', 'BraTS20_Training_052', 'BraTS20_Training_054',
       # 'BraTS20_Training_055', 'BraTS20_Training_060', 'BraTS20_Training_061', 'BraTS20_Training_062', 'BraTS20_Training_063',
       # 'BraTS20_Training_064', 'BraTS20_Training_065', 'BraTS20_Training_066', 'BraTS20_Training_067', 'BraTS20_Training_068',
       # 'BraTS20_Training_070', 'BraTS20_Training_072', 'BraTS20_Training_073', 'BraTS20_Training_074', 'BraTS20_Training_075',
       # 'BraTS20_Training_078', 'BraTS20_Training_079', 'BraTS20_Training_080', 'BraTS20_Training_081', 'BraTS20_Training_082',
       # 'BraTS20_Training_083', 'BraTS20_Training_084', 'BraTS20_Training_085', 'BraTS20_Training_086', 'BraTS20_Training_087',
       # 'BraTS20_Training_088', 'BraTS20_Training_091', 'BraTS20_Training_093', 'BraTS20_Training_094', 'BraTS20_Training_096',
       # 'BraTS20_Training_097', 'BraTS20_Training_098', 'BraTS20_Training_100', 'BraTS20_Training_101', 'BraTS20_Training_102',
       # 'BraTS20_Training_104', 'BraTS20_Training_108', 'BraTS20_Training_110', 'BraTS20_Training_111', 'BraTS20_Training_112',
       # 'BraTS20_Training_115', 'BraTS20_Training_116', 'BraTS20_Training_117', 'BraTS20_Training_119', 'BraTS20_Training_120',
       # 'BraTS20_Training_121', 'BraTS20_Training_122', 'BraTS20_Training_123', 'BraTS20_Training_125', 'BraTS20_Training_126',
       # 'BraTS20_Training_127', 'BraTS20_Training_128', 'BraTS20_Training_129', 'BraTS20_Training_130', 'BraTS20_Training_131',
       # 'BraTS20_Training_132', 'BraTS20_Training_133', 'BraTS20_Training_134', 'BraTS20_Training_135', 'BraTS20_Training_136',
       # 'BraTS20_Training_137', 'BraTS20_Training_138', 'BraTS20_Training_140', 'BraTS20_Training_141', 'BraTS20_Training_142',
       # 'BraTS20_Training_143', 'BraTS20_Training_144', 'BraTS20_Training_146', 'BraTS20_Training_148', 'BraTS20_Training_149',
       # 'BraTS20_Training_150', 'BraTS20_Training_153', 'BraTS20_Training_154', 'BraTS20_Training_155', 'BraTS20_Training_158',
       # 'BraTS20_Training_159', 'BraTS20_Training_160', 'BraTS20_Training_162', 'BraTS20_Training_163', 'BraTS20_Training_164',
       # 'BraTS20_Training_165', 'BraTS20_Training_166', 'BraTS20_Training_167', 'BraTS20_Training_168', 'BraTS20_Training_169',
       # 'BraTS20_Training_170', 'BraTS20_Training_171', 'BraTS20_Training_173', 'BraTS20_Training_174', 'BraTS20_Training_175',
       # 'BraTS20_Training_177', 'BraTS20_Training_178', 'BraTS20_Training_179', 'BraTS20_Training_180', 'BraTS20_Training_182',
       # 'BraTS20_Training_183', 'BraTS20_Training_184', 'BraTS20_Training_185', 'BraTS20_Training_186', 'BraTS20_Training_187',
       # 'BraTS20_Training_188', 'BraTS20_Training_189', 'BraTS20_Training_191', 'BraTS20_Training_192', 'BraTS20_Training_193',
       # 'BraTS20_Training_195', 'BraTS20_Training_197', 'BraTS20_Training_199', 'BraTS20_Training_200', 'BraTS20_Training_201',
       # 'BraTS20_Training_202', 'BraTS20_Training_203', 'BraTS20_Training_206', 'BraTS20_Training_207', 'BraTS20_Training_208',
       # 'BraTS20_Training_210', 'BraTS20_Training_211', 'BraTS20_Training_212', 'BraTS20_Training_213', 'BraTS20_Training_214',
       # 'BraTS20_Training_215', 'BraTS20_Training_216', 'BraTS20_Training_217', 'BraTS20_Training_218', 'BraTS20_Training_219',
       # 'BraTS20_Training_222', 'BraTS20_Training_223', 'BraTS20_Training_224', 'BraTS20_Training_225', 'BraTS20_Training_226',
       # 'BraTS20_Training_228', 'BraTS20_Training_229', 'BraTS20_Training_230', 'BraTS20_Training_231', 'BraTS20_Training_232',
       # 'BraTS20_Training_233', 'BraTS20_Training_236', 'BraTS20_Training_237', 'BraTS20_Training_238', 'BraTS20_Training_239',
       # 'BraTS20_Training_241', 'BraTS20_Training_243', 'BraTS20_Training_244', 'BraTS20_Training_246', 'BraTS20_Training_247',
       # 'BraTS20_Training_248', 'BraTS20_Training_249', 'BraTS20_Training_251', 'BraTS20_Training_252', 'BraTS20_Training_253',
       # 'BraTS20_Training_254', 'BraTS20_Training_255', 'BraTS20_Training_258', 'BraTS20_Training_259', 'BraTS20_Training_261',
       # 'BraTS20_Training_262', 'BraTS20_Training_263', 'BraTS20_Training_264', 'BraTS20_Training_265', 'BraTS20_Training_266',
       # 'BraTS20_Training_267', 'BraTS20_Training_268', 'BraTS20_Training_272', 'BraTS20_Training_273', 'BraTS20_Training_274',
       # 'BraTS20_Training_275', 'BraTS20_Training_276', 'BraTS20_Training_277', 'BraTS20_Training_278', 'BraTS20_Training_279',
       # 'BraTS20_Training_280', 'BraTS20_Training_283', 'BraTS20_Training_284', 'BraTS20_Training_285', 'BraTS20_Training_286',
       # 'BraTS20_Training_288', 'BraTS20_Training_290', 'BraTS20_Training_293', 'BraTS20_Training_294', 'BraTS20_Training_296',
       # 'BraTS20_Training_297', 'BraTS20_Training_298', 'BraTS20_Training_299', 'BraTS20_Training_300', 'BraTS20_Training_301',
       # 'BraTS20_Training_302', 'BraTS20_Training_303', 'BraTS20_Training_304', 'BraTS20_Training_306', 'BraTS20_Training_307',
       # 'BraTS20_Training_308', 'BraTS20_Training_309', 'BraTS20_Training_311', 'BraTS20_Training_312', 'BraTS20_Training_313',
       # 'BraTS20_Training_315', 'BraTS20_Training_316', 'BraTS20_Training_317', 'BraTS20_Training_318', 'BraTS20_Training_319',
       # 'BraTS20_Training_320', 'BraTS20_Training_321', 'BraTS20_Training_322', 'BraTS20_Training_324', 'BraTS20_Training_326',
       # 'BraTS20_Training_328', 'BraTS20_Training_329', 'BraTS20_Training_332', 'BraTS20_Training_334', 'BraTS20_Training_335',
       # 'BraTS20_Training_336', 'BraTS20_Training_338', 'BraTS20_Training_339', 'BraTS20_Training_340', 'BraTS20_Training_341',
       # 'BraTS20_Training_342', 'BraTS20_Training_343', 'BraTS20_Training_344', 'BraTS20_Training_345', 'BraTS20_Training_347',
       # 'BraTS20_Training_348', 'BraTS20_Training_349', 'BraTS20_Training_351', 'BraTS20_Training_353', 'BraTS20_Training_354',
       # 'BraTS20_Training_355', 'BraTS20_Training_356', 'BraTS20_Training_357', 'BraTS20_Training_358', 'BraTS20_Training_359',
       # 'BraTS20_Training_360', 'BraTS20_Training_363', 'BraTS20_Training_364', 'BraTS20_Training_365', 'BraTS20_Training_366',
       # 'BraTS20_Training_367', 'BraTS20_Training_368', 'BraTS20_Training_369', 'BraTS20_Training_370', 'BraTS20_Training_371',
       # 'BraTS20_Training_372', 'BraTS20_Training_373', 'BraTS20_Training_374', 'BraTS20_Training_375', 'BraTS20_Training_376',
       # 'BraTS20_Training_377', 'BraTS20_Training_378', 'BraTS20_Training_379', 'BraTS20_Training_380', 'BraTS20_Training_381',
       # 'BraTS20_Training_383', 'BraTS20_Training_384', 'BraTS20_Training_385', 'BraTS20_Training_386', 'BraTS20_Training_387',
       # 'BraTS20_Training_388', 'BraTS20_Training_390', 'BraTS20_Training_391', 'BraTS20_Training_392', 'BraTS20_Training_393',
       # 'BraTS20_Training_394', 'BraTS20_Training_395', 'BraTS20_Training_396', 'BraTS20_Training_398', 'BraTS20_Training_399',
       # 'BraTS20_Training_401', 'BraTS20_Training_403', 'BraTS20_Training_404', 'BraTS20_Training_405', 'BraTS20_Training_407',
       # 'BraTS20_Training_408', 'BraTS20_Training_409', 'BraTS20_Training_410', 'BraTS20_Training_411', 'BraTS20_Training_412',
       # 'BraTS20_Training_413', 'BraTS20_Training_414', 'BraTS20_Training_415', 'BraTS20_Training_417', 'BraTS20_Training_418',
       # 'BraTS20_Training_419', 'BraTS20_Training_420', 'BraTS20_Training_421', 'BraTS20_Training_422', 'BraTS20_Training_423',
       # 'BraTS20_Training_424', 'BraTS20_Training_426', 'BraTS20_Training_428', 'BraTS20_Training_429', 'BraTS20_Training_430',
       # 'BraTS20_Training_431', 'BraTS20_Training_433', 'BraTS20_Training_434', 'BraTS20_Training_435', 'BraTS20_Training_436',
       # 'BraTS20_Training_437', 'BraTS20_Training_438', 'BraTS20_Training_439', 'BraTS20_Training_441', 'BraTS20_Training_442',
       # 'BraTS20_Training_443', 'BraTS20_Training_444', 'BraTS20_Training_445', 'BraTS20_Training_446', 'BraTS20_Training_449',
       # 'BraTS20_Training_451', 'BraTS20_Training_452', 'BraTS20_Training_453', 'BraTS20_Training_454', 'BraTS20_Training_455',
       # 'BraTS20_Training_457', 'BraTS20_Training_458', 'BraTS20_Training_459', 'BraTS20_Training_460', 'BraTS20_Training_463',
       # 'BraTS20_Training_464', 'BraTS20_Training_466', 'BraTS20_Training_467', 'BraTS20_Training_468', 'BraTS20_Training_469',
       # 'BraTS20_Training_470', 'BraTS20_Training_472', 'BraTS20_Training_475', 'BraTS20_Training_477', 'BraTS20_Training_478',
       # 'BraTS20_Training_481', 'BraTS20_Training_482', 'BraTS20_Training_483','BraTS20_Training_400', 'BraTS20_Training_402',
       # 'BraTS20_Training_406', 'BraTS20_Training_416', 'BraTS20_Training_427', 'BraTS20_Training_440', 'BraTS20_Training_447',
       # 'BraTS20_Training_448', 'BraTS20_Training_456', 'BraTS20_Training_461', 'BraTS20_Training_462', 'BraTS20_Training_465',
       # 'BraTS20_Training_471', 'BraTS20_Training_473', 'BraTS20_Training_474', 'BraTS20_Training_476', 'BraTS20_Training_479',
       # 'BraTS20_Training_480', 'BraTS20_Training_484'])
       #      splits[self.fold]['val']=np.array(['BraTS20_Training_011', 'BraTS20_Training_012', 'BraTS20_Training_018', 'BraTS20_Training_020', 'BraTS20_Training_021',
       # 'BraTS20_Training_028', 'BraTS20_Training_029', 'BraTS20_Training_032', 'BraTS20_Training_034', 'BraTS20_Training_036',
       # 'BraTS20_Training_041', 'BraTS20_Training_047', 'BraTS20_Training_049', 'BraTS20_Training_053', 'BraTS20_Training_056',
       # 'BraTS20_Training_057', 'BraTS20_Training_069', 'BraTS20_Training_071', 'BraTS20_Training_089', 'BraTS20_Training_090',
       # 'BraTS20_Training_092', 'BraTS20_Training_095', 'BraTS20_Training_103', 'BraTS20_Training_105', 'BraTS20_Training_106',
       # 'BraTS20_Training_107', 'BraTS20_Training_109', 'BraTS20_Training_118', 'BraTS20_Training_145', 'BraTS20_Training_147',
       # 'BraTS20_Training_156', 'BraTS20_Training_161', 'BraTS20_Training_172', 'BraTS20_Training_176', 'BraTS20_Training_181',
       # 'BraTS20_Training_194', 'BraTS20_Training_196', 'BraTS20_Training_198', 'BraTS20_Training_204', 'BraTS20_Training_205',
       # 'BraTS20_Training_209', 'BraTS20_Training_220', 'BraTS20_Training_221', 'BraTS20_Training_227', 'BraTS20_Training_234',
       # 'BraTS20_Training_235', 'BraTS20_Training_245', 'BraTS20_Training_250', 'BraTS20_Training_256', 'BraTS20_Training_257',
       # 'BraTS20_Training_260', 'BraTS20_Training_269', 'BraTS20_Training_270', 'BraTS20_Training_271', 'BraTS20_Training_281',
       # 'BraTS20_Training_282', 'BraTS20_Training_287', 'BraTS20_Training_289', 'BraTS20_Training_291', 'BraTS20_Training_292',
       # 'BraTS20_Training_310', 'BraTS20_Training_314', 'BraTS20_Training_323', 'BraTS20_Training_327', 'BraTS20_Training_330',
       # 'BraTS20_Training_333', 'BraTS20_Training_337', 'BraTS20_Training_346', 'BraTS20_Training_350', 'BraTS20_Training_352',
       # 'BraTS20_Training_361', 'BraTS20_Training_382', 'BraTS20_Training_397'])
            if self.fold < len(splits):
                tr_keys = splits[self.fold]['train']
                val_keys = splits[self.fold]['val']
                self.print_to_log_file("This split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))
            else:
                self.print_to_log_file("INFO: You requested fold %d for training but splits "
                                       "contain only %d folds. I am now creating a "
                                       "random (but seeded) 80:20 split!" % (self.fold, len(splits)))
                # if we request a fold that is not in the split file, create a random 80:20 split
                rnd = np.random.RandomState(seed=12345 + self.fold)
                keys = np.sort(list(self.dataset.keys()))
                idx_tr = rnd.choice(len(keys), int(len(keys) * 0.8), replace=False)
                idx_val = [i for i in range(len(keys)) if i not in idx_tr]
                tr_keys = [keys[i] for i in idx_tr]
                val_keys = [keys[i] for i in idx_val]
                self.print_to_log_file("This random 80:20 split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))

        tr_keys.sort()
        val_keys.sort()
        self.dataset_tr = OrderedDict()
        for i in tr_keys:
            # print(i)
            # print(self.dataset_tr.keys())
            # print(self.dataset.keys())
            self.dataset_tr[i] = self.dataset[i]
        self.dataset_val = OrderedDict()
        for i in val_keys:
            self.dataset_val[i] = self.dataset[i]

    def setup_DA_params(self):
        """
        - we increase roation angle from [-15, 15] to [-30, 30]
        - scale range is now (0.7, 1.4), was (0.85, 1.25)
        - we don't do elastic deformation anymore
        :return:
        """

        self.deep_supervision_scales = [[1, 1, 1]] + list(list(i) for i in 1 / np.cumprod(
            np.vstack(self.net_num_pool_op_kernel_sizes), axis=0))[:-1]

        if self.threeD:
            self.data_aug_params = default_3D_augmentation_params
            self.data_aug_params['rotation_x'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_y'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_z'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            if self.do_dummy_2D_aug:
                self.data_aug_params["dummy_2D"] = True
                self.print_to_log_file("Using dummy2d data augmentation")
                self.data_aug_params["elastic_deform_alpha"] = \
                    default_2D_augmentation_params["elastic_deform_alpha"]
                self.data_aug_params["elastic_deform_sigma"] = \
                    default_2D_augmentation_params["elastic_deform_sigma"]
                self.data_aug_params["rotation_x"] = default_2D_augmentation_params["rotation_x"]
        else:
            self.do_dummy_2D_aug = False
            if max(self.patch_size) / min(self.patch_size) > 1.5:
                default_2D_augmentation_params['rotation_x'] = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
            self.data_aug_params = default_2D_augmentation_params
        self.data_aug_params["mask_was_used_for_normalization"] = self.use_mask_for_norm

        if self.do_dummy_2D_aug:
            self.basic_generator_patch_size = get_patch_size(self.patch_size[1:],
                                                             self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])
            self.basic_generator_patch_size = np.array([self.patch_size[0]] + list(self.basic_generator_patch_size))
            patch_size_for_spatialtransform = self.patch_size[1:]
        else:
            self.basic_generator_patch_size = get_patch_size(self.patch_size, self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])
            patch_size_for_spatialtransform = self.patch_size

        self.data_aug_params["scale_range"] = (0.7, 1.4)
        self.data_aug_params["do_elastic"] = False
        self.data_aug_params['selected_seg_channels'] = [0]
        self.data_aug_params['patch_size_for_spatialtransform'] = patch_size_for_spatialtransform

        self.data_aug_params["num_cached_per_thread"] = 2

    def maybe_update_lr(self, epoch=None):
        """
        if epoch is not None we overwrite epoch. Else we use epoch = self.epoch + 1
        (maybe_update_lr is called in on_epoch_end which is called before epoch is incremented.
        herefore we need to do +1 here)
        :param epoch:
        :return:
        """
        if epoch is None:
            ep = self.epoch + 1
        else:
            ep = epoch
        self.optimizer.param_groups[0]['lr'] = poly_lr(ep, self.max_num_epochs, self.initial_lr, 0.9)
        self.print_to_log_file("lr:", np.round(self.optimizer.param_groups[0]['lr'], decimals=6))

    def on_epoch_end(self):
        """
        overwrite patient-based early stopping. Always run to 1000 epochs
        :return:
        """
        super().on_epoch_end()
        continue_training = self.epoch < self.max_num_epochs

        # it can rarely happen that the momentum of nnUNetTrainerV2 is too high for some dataset. If at epoch 100 the
        # estimated validation Dice is still 0 then we reduce the momentum from 0.99 to 0.95
        if self.epoch == 100:
            if self.all_val_eval_metrics[-1] == 0:
                self.optimizer.param_groups[0]["momentum"] = 0.95
                self.network.apply(InitWeights_He(1e-2))
                self.print_to_log_file("At epoch 100, the mean foreground Dice was 0. This can be caused by a too "
                                       "high momentum. High momentum (0.99) is good for datasets where it works, but "
                                       "sometimes causes issues such as this one. Momentum has now been reduced to "
                                       "0.95 and network weights have been reinitialized")
        return continue_training

    def run_training(self):
        """
        if we run with -c then we need to set the correct lr for the first epoch, otherwise it will run the first
        continued epoch with self.initial_lr
        we also need to make sure deep supervision in the network is enabled for training, thus the wrapper
        :return:
        """
        self.maybe_update_lr(self.epoch)  # if we dont overwrite epoch then self.epoch+1 is used which is not what we
        # want at the start of the training
        ds = self.network.do_ds
        if self.deep_supervision:
            self.network.do_ds = True
        else:
            self.network.do_ds = False
        ret = super().run_training()
        self.network.do_ds = ds
        return ret