# ------------------------------------------------------------------------
# UTrans
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from UTrans.network_architecture import CNNBackbone
from nnunet.network_architecture.neural_network import SegmentationNetwork
from UTrans.network_architecture.DeTrans.DeformableTrans import DeformableTransformer
from UTrans.network_architecture.DeTrans.position_encoding import build_position_encoding
import math

from einops import rearrange


class Conv3d_wd(nn.Conv3d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,1,1), padding=(0,0,0), dilation=(1,1,1), groups=1, bias=False):
        super(Conv3d_wd, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True).mean(dim=4, keepdim=True)
        weight = weight - weight_mean
        # std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1, 1) + 1e-5
        std = torch.sqrt(torch.var(weight.view(weight.size(0), -1), dim=1) + 1e-12).view(-1, 1, 1, 1, 1)
        weight = weight / std.expand_as(weight)
        return F.conv3d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def conv3x3x3(in_planes, out_planes, kernel_size, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), groups=1, bias=False, weight_std=False):
    "3x3x3 convolution with padding"
    if weight_std:
        return Conv3d_wd(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
    else:
        return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)


def Norm_layer(norm_cfg, inplanes):

    if norm_cfg == 'BN':
        out = nn.BatchNorm3d(inplanes)
    elif norm_cfg == 'SyncBN':
        out = nn.SyncBatchNorm(inplanes)
    elif norm_cfg == 'GN':
        out = nn.GroupNorm(16, inplanes)
    elif norm_cfg == 'IN':
        out = nn.InstanceNorm3d(inplanes,affine=True)

    return out


def Activation_layer(activation_cfg, inplace=True):

    if activation_cfg == 'ReLU':
        out = nn.ReLU(inplace=inplace)
    elif activation_cfg == 'LeakyReLU':
        out = nn.LeakyReLU(negative_slope=1e-2, inplace=inplace)

    return out


class Conv3dBlock(nn.Module):
    def __init__(self,in_channels,out_channels,norm_cfg,activation_cfg,kernel_size,stride=(1, 1, 1),padding=(0, 0, 0),dilation=(1, 1, 1),bias=False,weight_std=False):
        super(Conv3dBlock,self).__init__()
        self.conv = conv3x3x3(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, weight_std=weight_std)
        self.norm = Norm_layer(norm_cfg, out_channels)
        self.nonlin = Activation_layer(activation_cfg, inplace=True)
    def forward(self,x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.nonlin(x)
        return x

class ResBlock(nn.Module):

    def __init__(self, inplanes, planes, norm_cfg, activation_cfg, weight_std=False):
        super(ResBlock, self).__init__()
        self.resconv1 = Conv3dBlock(inplanes, planes, norm_cfg, activation_cfg, kernel_size=3, stride=1, padding=1, bias=False, weight_std=weight_std)
        self.resconv2 = Conv3dBlock(planes, planes, norm_cfg, activation_cfg, kernel_size=3, stride=1, padding=1, bias=False, weight_std=weight_std)

    def forward(self, x):
        residual = x

        out = self.resconv1(x)
        out = self.resconv2(out)
        out = out + residual

        return out


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Setr3d_Module(nn.Module):
    def __init__(self, norm_cfg='BN', activation_cfg='ReLU', img_size=None, num_classes=None, weight_std=False):
        super(Setr3d_Module, self).__init__()

        self.MODEL_NUM_CLASSES = num_classes

        self.in_dim_ = 4096
        self.d_model = 1024 
        # self.filters = [128, 256, 512, 1024]
        self.filters = [256, 256, 256, 1024]
        d_model = self.d_model

        self.linear_projection = nn.Linear(self.in_dim_, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model, dropout=0., max_len=5000)

        nheads = 16
        encodlayer = nn.TransformerEncoderLayer(d_model = d_model, nhead = nheads, dim_feedforward=4*self.d_model, dropout=0.1, activation='relu')
        self.transformers_0 = nn.TransformerEncoder(encoder_layer=encodlayer, num_layers=10)
        
        encodlayer = nn.TransformerEncoderLayer(d_model = d_model, nhead = nheads, dim_feedforward=4*self.d_model, dropout=0.1, activation='relu')
        self.transformers_1 = nn.TransformerEncoder(encoder_layer=encodlayer, num_layers=5)
        
        encodlayer = nn.TransformerEncoderLayer(d_model = d_model, nhead = nheads, dim_feedforward=4*self.d_model, dropout=0.1, activation='relu')
        self.transformers_2 = nn.TransformerEncoder(encoder_layer=encodlayer, num_layers=5)
        
        encodlayer = nn.TransformerEncoderLayer(d_model = d_model, nhead = nheads, dim_feedforward=4*self.d_model, dropout=0.1, activation='relu')
        self.transformers_3 = nn.TransformerEncoder(encoder_layer=encodlayer, num_layers=4)

        self.ds3_cls_conv = nn.Sequential(nn.Conv3d(d_model, self.MODEL_NUM_CLASSES, kernel_size=1), nn.Upsample(scale_factor=( 2,1,1)))
        self.ds2_cls_conv = nn.Sequential(nn.Conv3d(d_model, self.MODEL_NUM_CLASSES, kernel_size=1), nn.Upsample(scale_factor=( 4,2,2)))
        self.ds1_cls_conv = nn.Sequential(nn.Conv3d(d_model, self.MODEL_NUM_CLASSES, kernel_size=1), nn.Upsample(scale_factor=(8,4,4)))
        self.ds0_cls_conv = nn.Sequential(nn.Conv3d(d_model, self.MODEL_NUM_CLASSES, kernel_size=1), nn.Upsample(scale_factor=(16,8,8)))


        # self.transposeconv_stage3 = nn.Sequential(nn.ConvTranspose3d(d_model, self.filters[3], kernel_size=(2,2,2), stride=(2,2,2), bias=False),nn.ReLU(),
        #                                             nn.Conv3d(self.filters[3], self.filters[3], kernel_size=3, padding=1), nn.ReLU())
        # self.transposeconv_stage2 = nn.Sequential(nn.ConvTranspose3d(self.filters[3], self.filters[2], kernel_size=(2,2,2), stride=(2,2,2), bias=False),nn.ReLU(),
        #                                             nn.Conv3d(self.filters[2], self.filters[2], kernel_size=3, padding=1), nn.ReLU())
        # self.transposeconv_stage1 = nn.Sequential(nn.ConvTranspose3d(self.filters[2], self.filters[1], kernel_size=(2,2,2), stride=(2,2,2), bias=False),nn.ReLU(),
        #                                             nn.Conv3d(self.filters[1], self.filters[1], kernel_size=3, padding=1), nn.ReLU())
        # self.transposeconv_stage0 = nn.Sequential(nn.ConvTranspose3d(self.filters[1], self.filters[0], kernel_size=(2,2,2), stride=(2,2,2), bias=False),nn.ReLU(),
        #                                             nn.Conv3d(self.filters[0], self.filters[0], kernel_size=3, padding=1), nn.ReLU())


        # self.transposeconv_stage3 = nn.Sequential(nn.ConvTranspose3d(d_model, self.filters[3], kernel_size=(2,2,2), stride=(2,2,2), bias=False), 
        #                                             ResBlock(self.filters[3], self.filters[3], norm_cfg, activation_cfg, weight_std=weight_std))
        # self.transposeconv_stage2 = nn.Sequential(nn.ConvTranspose3d(self.filters[3], self.filters[2], kernel_size=(2,2,2), stride=(2,2,2), bias=False), 
        #                                             ResBlock(self.filters[2], self.filters[2], norm_cfg, activation_cfg, weight_std=weight_std))
        # self.transposeconv_stage1 = nn.Sequential(nn.ConvTranspose3d(self.filters[2], self.filters[1], kernel_size=(2,2,2), stride=(2,2,2), bias=False), 
        #                                             ResBlock(self.filters[1], self.filters[1], norm_cfg, activation_cfg, weight_std=weight_std))
        # self.transposeconv_stage0 = nn.Sequential(nn.ConvTranspose3d(self.filters[1], self.filters[0], kernel_size=(2,2,2), stride=(2,2,2), bias=False), 
        #                                             ResBlock(self.filters[0], self.filters[0], norm_cfg, activation_cfg, weight_std=weight_std))

        self.transposeconv_stage3 = nn.Sequential(nn.Conv3d(d_model, self.filters[3], kernel_size=3, stride=1, padding=1), 
                                                nn.Upsample(scale_factor=2))
        self.transposeconv_stage2 = nn.Sequential(nn.Conv3d(self.filters[3], self.filters[2],  kernel_size=3, stride=1, padding=1),
                                                nn.Upsample(scale_factor=2))
        self.transposeconv_stage1 = nn.Sequential(nn.Conv3d(self.filters[2], self.filters[1],  kernel_size=3, stride=1, padding=1),
                                                nn.Upsample(scale_factor=2))
        self.transposeconv_stage0 = nn.Sequential(nn.Conv3d(self.filters[1], self.filters[0],  kernel_size=3, stride=1, padding=1),
                                                nn.Upsample(scale_factor=2))


        self.cls_conv = nn.Conv3d(self.filters[0], self.MODEL_NUM_CLASSES, kernel_size=1)

        for m in self.modules():
            if isinstance(m, (nn.Conv3d, Conv3d_wd, nn.ConvTranspose3d, nn.Linear)):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, (nn.BatchNorm3d, nn.SyncBatchNorm, nn.InstanceNorm3d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def posi_mask(self, x):

        x_fea = []
        x_posemb = []
        masks = []
        for lvl, fea in enumerate(x):
            if lvl > 1:
                x_fea.append(fea)
                x_posemb.append(self.position_embed(fea))
                masks.append(torch.zeros((fea.shape[0], fea.shape[2], fea.shape[3], fea.shape[4]), dtype=torch.bool).cuda())

        return x_fea, masks, x_posemb


    def forward(self, inputs):
        # # %%%%%%%%%%%%% SeTr
        bs, c, d, w, h = inputs.shape
        seq = int(d*h*w*c/(16**3))

        # Reshape, project and position
        # inputs = torch.reshape(inputs, (bs, seq, self.in_dim_))
        inputs = rearrange(inputs, "b c (x o) (y p) (z q) -> b c x o y p z q", x=d//16, y=w//16, z=h//16, o=16,p=16,q=16)
        inputs = rearrange(inputs, "b c x o y p z q -> b c x y z o p q")
        inputs = rearrange(inputs, "b c x y z o p q -> b (c x y z) (o p q)")
        inputs = self.linear_projection(inputs)

        # Transformer : permute for pytorch trans (seq, bs, d_model) and position encode
        # inputs = inputs.permute((1,0,2))
        inputs = self.pos_encoder(inputs)
        inputs = rearrange(inputs, 'n s d -> s n d')

        print("inputs",inputs.shape)
        exit(0)

        skip_0 = self.transformers_0(inputs)
        del inputs
        skip_1 = self.transformers_1(skip_0)
        skip_2 = self.transformers_2(skip_1)
        skip_3 = self.transformers_3(skip_2)

        # Deep Supervision
        # ds3 = self.ds3_cls_conv(torch.reshape(rearrange(skip_0, 's n d -> n s d'), (bs, c*self.d_model, int(d/16), int(w/16), int(h/16))))
        ds3 = rearrange(skip_0, 's n d -> n s d')
        print(ds3.shape)
        ds3 = rearrange(ds3, "b c (x y z) -> b c x y z", x=int(d/16), y=int(w/16), z=int(h/16))
        ds3 = self.ds3_cls_conv(ds3)
        del skip_0
        # ds2 = self.ds2_cls_conv(torch.reshape(rearrange(skip_1, 's n d -> n s d'), (bs, c*self.d_model, int(d/16), int(w/16), int(h/16))))
        ds2 = rearrange(skip_1, 's n d -> n s d')
        ds2 = rearrange(ds2, "b c (x y z) -> b c x y z", x=int(d/16), y=int(w/16), z=int(h/16))
        ds2 = self.ds2_cls_conv(ds2)
        del skip_1
        # ds1 = self.ds1_cls_conv(torch.reshape(rearrange(skip_2, 's n d -> n s d'), (bs, c*self.d_model, int(d/16), int(w/16), int(h/16))))
        ds1 = rearrange(skip_2, 's n d -> n s d')
        ds1 = rearrange(ds1, "b c (x y z) -> b c x y z", x=int(d/16), y=int(w/16), z=int(h/16))
        ds1 = self.ds1_cls_conv(ds1)
        del skip_2
        # ds0 = self.ds0_cls_conv(torch.reshape(rearrange(skip_3, 's n d -> n s d'), (bs, c*self.d_model, int(d/16), int(w/16), int(h/16))))
        ds0 = rearrange(skip_3, 's n d -> n s d')
        ds0 = rearrange(ds0, "b c (x y z) -> b c x y z", x=int(d/16), y=int(w/16), z=int(h/16))
        ds0 = self.ds0_cls_conv(ds0)
        # Deconv
        # result = self.transposeconv_stage3(torch.reshape(rearrange(skip_3, 's n d -> n s d'), (bs, c*self.d_model, int(d/16), int(w/16), int(h/16))))
        result = rearrange(skip_3, 's n d -> n s d')
        result = rearrange(result, "b c (x y z) -> b c x y z", x=int(d/16), y=int(w/16), z=int(h/16))
        result = self.transposeconv_stage3(result)
        del skip_3
        result = self.transposeconv_stage2(result)
        result = self.transposeconv_stage1(result)
        result = self.transposeconv_stage0(result)

        # Prediction
        result = self.cls_conv(result)

        # print("result", result.shape)
        # print("ds0", ds0.shape)
        # print("ds1", ds1.shape)
        # print("ds2", ds2.shape)
        # print("ds3", ds3.shape)
        # exit(0)
        return [result, ds0, ds1, ds2, ds3]


class Setr3d(SegmentationNetwork):
    """
    ResTran-3D Unet
    """
    def __init__(self, norm_cfg='BN', activation_cfg='ReLU', img_size=None, num_classes=None, weight_std=False, deep_supervision=False):
        super().__init__()
        self.do_ds = False
        self.Setr3d_Module = Setr3d_Module(norm_cfg, activation_cfg, img_size, num_classes, weight_std) # Setr3d_Module

        if weight_std==False:
            self.conv_op = nn.Conv3d
        else:
            self.conv_op = Conv3d_wd
        if norm_cfg=='BN':
            self.norm_op = nn.BatchNorm3d
        if norm_cfg=='SyncBN':
            self.norm_op = nn.SyncBatchNorm
        if norm_cfg=='GN':
            self.norm_op = nn.GroupNorm
        if norm_cfg=='IN':
            self.norm_op = nn.InstanceNorm3d
        self.dropout_op = nn.Dropout3d
        self.num_classes = num_classes
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision

    def forward(self, x):
        seg_output = self.Setr3d_Module(x)
        if self._deep_supervision and self.do_ds:
            return seg_output
        else:
            return seg_output[0]
