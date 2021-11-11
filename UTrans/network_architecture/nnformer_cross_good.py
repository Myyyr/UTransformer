from einops import rearrange
from copy import deepcopy
from nnunet.utilities.nd_softmax import softmax_helper
from torch import nn
import torch
import numpy as np
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
import torch.nn.functional


import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_3tuple, trunc_normal_


class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, S, H, W, C = x.shape
    x = x.view(B, S // window_size, window_size, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, S, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (S * H * W / window_size / window_size / window_size))
    x = windows.view(B, S // window_size, H // window_size, W // window_size, window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, S, H, W, -1)
    return x



class WindowCrossAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., channel_scale=2):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.channel_scale = channel_scale

        # define a parameter table of relative position bias
        # self.relative_position_bias_table = nn.Parameter(
        #     torch.zeros((2 * window_size[0]*2 - 1) * (2 * window_size[1]*2 - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # # get pair-wise relative position index for each token inside the window
        # coords_h = torch.arange(self.window_size[0]).repeat_interleave(2)
        # coords_w = torch.arange(self.window_size[1]).repeat_interleave(2)
        # coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww


        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0]*2 - 1) * (2 * window_size[1]*2 - 1) * (2 * window_size[2]*2 - 1),
                        num_heads))  

        # get pair-wise relative position index for each token inside the window
        coords_s = torch.arange(self.window_size[0]).repeat_interleave(2)
        coords_h = torch.arange(self.window_size[1]).repeat_interleave(2)
        coords_w = torch.arange(self.window_size[2]).repeat_interleave(2)
        coords = torch.stack(torch.meshgrid([coords_s, coords_h, coords_w]))
        
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords[:,:,::4] # take slices in one axis
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        # relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        # relative_coords[:, :, 1] += self.window_size[1] - 1
        # relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1

        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= 3 * self.window_size[1] - 1
        relative_coords[:, :, 1] *= 2 * self.window_size[1] - 1


        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.proj_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj_q = nn.Linear(dim // channel_scale, dim, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim // channel_scale)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, kv, q, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # B_ = num_windows*B
        B_, N, C = kv.shape

        qB_, qN, qC = q.shape
        
        kv = self.proj_kv(kv).reshape(B_, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q = self.proj_q(q).reshape(qB_, qN, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        k, v = kv[0], kv[1] # make torchscript happy (cannot use tensor as tuple)
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            (self.window_size[0] * 2) * (self.window_size[1] * 2), self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)
    
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(qB_, qN, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops




class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1),
                        num_heads))  

        # get pair-wise relative position index for each token inside the window
        coords_s = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid([coords_s, coords_h, coords_w]))  
        coords_flatten = torch.flatten(coords, 1)  
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        
        relative_coords[:, :, 0] *= 3 * self.window_size[1] - 1
        relative_coords[:, :, 1] *= 2 * self.window_size[1] - 1

        relative_position_index = relative_coords.sum(-1)  
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """ Forward function.

        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        
        qkv = self.qkv(x)
        
        qkv=qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2], -1)  
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttentionBlock(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, skip_connection_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, residual_patch_expand=True, channel_scale=2):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.skip_connection_resolution = skip_connection_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.window_size_skip_co = self.window_size * 2
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.channel_scale = channel_scale
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.proj_shortcut = nn.Linear(dim, dim // self.channel_scale)
        # self.upsample_shortcut = nn.UpsamplingBilinear2d(scale_factor=(2,2))
        self.expand = Patch_Expanding(dim)

        self.residual_patch_expand = residual_patch_expand
        
        self.norm1 = norm_layer(dim)
        self.attn = WindowCrossAttention(
            dim, window_size=to_3tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            channel_scale=channel_scale)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim // self.channel_scale)
        mlp_hidden_dim = int(dim // self.channel_scale * mlp_ratio)
        self.mlp = Mlp(in_features=dim // self.channel_scale, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

 

    def forward(self, x, x_downsampled, padwh):
        B, L, C = x.shape
        S, H, W = self.input_resolution

        S_d, H_d, W_d = self.skip_connection_resolution
        B_d, L_d, C_d = x_downsampled.shape

        assert L == S * H * W, "input feature has wrong size"
        assert L_d == S_d * H_d * W_d, "cross input feature has wrong size"

        shortcut, Sh, Wh, Ww = self.expand(x, S, H, W, padwh)
        # if self.residual_patch_expand:
        #     shortcut, Sh, Wh, Ww = self.expand(x, S, H, W, padwh)
        # else:
        #     shortcut = self.proj_shortcut(x)
        #     shortcut = shortcut.view(B, H, W, C_d).permute(0,3,1,2)
        #     shortcut = self.upsample_shortcut(shortcut).permute(0,2,3,1)
        #     if padwh[0] != 0 or padwh[1] != 0:
        #         shortcut = shortcut[:,:(shortcut.shape[1]-padwh[1]),:(shortcut.shape[2]-padwh[0]),:]
        #         shortcut = shortcut.contiguous()
        #     shortcut = shortcut.view(B, L_d, C_d)

        
        x = self.norm1(x)
        x = x.view(B, S, H, W, C)

        #x_downsampled = self.norm1(x_downsampled)
        x_downsampled = x_downsampled.view(B_d, S_d, H_d, W_d, C_d)
        

        # Pad x
        # pad_l = pad_t = 0
        # pad_r = (self.window_size - W % self.window_size) % self.window_size
        # pad_b = (self.window_size - H % self.window_size) % self.window_size
        # x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))

        # _, Hp, Wp, _ = x.shape

        # pad feature maps to multiples of window size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        pad_g = (self.window_size - S % self.window_size) % self.window_size

        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b, 0, pad_g))  
        _, Sp, Hp, Wp, _ = x.shape


        # Pad x_downsampled
        d_pad_l = d_pad_t = 0
        d_pad_r = (self.window_size_skip_co - W_d % self.window_size_skip_co) % self.window_size_skip_co
        d_pad_b = (self.window_size_skip_co - H_d % self.window_size_skip_co) % self.window_size_skip_co
        x_downsampled = F.pad(x_downsampled, (0, 0, d_pad_l, d_pad_r, d_pad_t, d_pad_b))

        _, Hp_d, Wp_d, _ = x_downsampled.shape
        

        # partition windows
        # x_windows = window_partition(x, self.window_size)  # nW*B, window_size, window_size, C
        # x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        x_windows = window_partition(x, self.window_size)  
        x_windows = x_windows.view(-1, self.window_size * self.window_size * self.window_size,
                                   C) 

        # partition windows
        x_downsampled_windows = window_partition(x_downsampled, self.window_size_skip_co)  # nW*B, window_size, window_size, C
        x_downsampled_windows = x_downsampled_windows.view(-1, (self.window_size_skip_co) * (self.window_size_skip_co)* (self.window_size_skip_co), C_d)  # nW*B, window_size*window_size, C


        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, x_downsampled_windows, mask=None)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size_skip_co, self.window_size_skip_co, self.window_size_skip_co, C_d)
        x = window_reverse(attn_windows, self.window_size_skip_co, Sp, Hp_d, Wp_d)  # B H' W' C
            
        # if d_pad_r > 0 or d_pad_b > 0:
        #     x = x[:, :H_d, :W_d, :].contiguous()
        # x = x.view(B_d, H_d * W_d, C_d)

        if pad_r > 0 or pad_b > 0 or pad_g > 0:
            x = x[:, :S_d, :H_d, :W_d, :].contiguous()
        x = x.view(B, S_d * H_d * W_d, C_d)


        # FFN
        x = self.drop_path(x) + shortcut
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x, S_d, H_d, W_d

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        # H, W = self.input_resolution
        # # norm1
        # flops += self.dim * H * W
        # # W-MSA/SW-MSA
        # nW = H * W / self.window_size / self.window_size
        # flops += nW * self.attn.flops(self.window_size * self.window_size)
        # # mlp
        # flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # # norm2
        # flops += self.dim * H * W
        return flops




class SwinTransformerBlock(nn.Module):
    """ Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_3tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask_matrix):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.
        """
        B, L, C = x.shape
        S, H, W = self.input_resolution

        print("\n~~~~~~~~~", "SHW", S,H,W)
        print("~~~~~~~~~", "S*H*W", S*H*W)
        print("~~~~~~~~~", "x", x.shape)

        
        assert L == S * H * W, "input feature has wrong size"
        
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, S, H, W, C)

        # pad feature maps to multiples of window size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        pad_g = (self.window_size - S % self.window_size) % self.window_size

        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b, 0, pad_g))  
        _, Sp, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size,-self.shift_size), dims=(1, 2,3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  
        x_windows = x_windows.view(-1, self.window_size * self.window_size * self.window_size,
                                   C)  
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Sp, Hp, Wp)  

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size, self.shift_size), dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0 or pad_g > 0:
            x = x[:, :S, :H, :W, :].contiguous()

        x = x.view(B, S * H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    """ Patch Merging Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Conv3d(dim,dim*2,kernel_size=2,stride=2)
        self.norm = norm_layer(dim)

    def forward(self, x, S, H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W * S, "input feature has wrong size"

        x = x.view(B, S, H, W, C)

        
        x = F.gelu(x)
        x = self.norm(x)
        x=x.permute(0,4,1,2,3)
        x=self.reduction(x)
        x=x.permute(0,2,3,4,1).view(B,-1,2*C)
        return x, [S % 2, W % 2, H % 2]
    
class Patch_Expanding(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.norm = norm_layer(dim)
        self.up=nn.ConvTranspose3d(dim,dim//2,2,2)
    def forward(self, x, S, H, W, pad=None):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        
        print('\n---->x', x.shape,'\n')
        B, L, C = x.shape
        print('\n---->SHW', S, H, W,'\n')
        print('\n---->S*H*W', S*H*W,'\n')
        # exit(0)
        assert L == H * W * S, "input feature has wrong size"

        x = x.view(B, S, H, W, C)

       
        
        x = self.norm(x)
        x=x.permute(0,4,1,2,3)
        x = self.up(x)
        x=x.permute(0,2,3,4,1)
        Ws, Wh, Ww = x.size(1), x.size(2), x.size(3)
        x=x.view(B,-1,C//2)
       
        return x, Ws, Wh, Ww


class BasicLayer_up_Xattn(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, skip_connection_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False,
                 use_cross_attention=False, residual_patch_expand=True, channel_scale=2):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.skip_connection_resolution = skip_connection_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.use_cross_attention = use_cross_attention
        self.residual_patch_expand = True #residual_patch_expand
        self.channel_scale = channel_scale

        # build blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            if i == 0:
                layer = CrossAttentionBlock(dim=dim, input_resolution=input_resolution,
                                            skip_connection_resolution=skip_connection_resolution,
                                            num_heads=num_heads, window_size=window_size,
                                            shift_size=0 if (i % 2 == 0) else window_size // 2,
                                            mlp_ratio=mlp_ratio,
                                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                                            drop=drop, attn_drop=attn_drop,
                                            drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                            norm_layer=norm_layer,
                                            residual_patch_expand=residual_patch_expand,
                                            channel_scale=self.channel_scale)
            else:
                layer = SwinTransformerBlock(dim=dim // 2, input_resolution=[x * 2 for x in input_resolution],
                                             num_heads=num_heads, window_size=window_size,
                                             shift_size=0 if (i % 2 == 0) else window_size // 2,
                                             mlp_ratio=mlp_ratio,
                                             qkv_bias=qkv_bias, qk_scale=qk_scale,
                                             drop=drop, attn_drop=attn_drop,
                                             drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                             norm_layer=norm_layer)
            self.blocks.append(layer)

        # patch merging layer
        if upsample is not None:
            self.upsample = PatchExpand(input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x, x_downsampled, S, H, W, S_d, H_d, W_d, padwh):
        Sp = int(np.ceil(S_d / self.window_size)) * self.window_size
        Hp = int(np.ceil(H_d / self.window_size)) * self.window_size
        Wp = int(np.ceil(W_d / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Sp, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        s_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        # cnt = 0
        # for h in h_slices:
        #     for w in w_slices:
        #         img_mask[:, h, w, :] = cnt
        #         cnt += 1
        cnt = 0
        for s in s_slices:
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, s, h, w, :] = cnt
                    cnt += 1


        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for inx, blk in enumerate(self.blocks):
            blk.input_resolution = (S, H, W)
            blk.skip_connection_resolution = (S_d, H_d, W_d)
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                if inx == 0:
                    x, S, H, W = blk(x, x_downsampled, padwh)
                else:
                    x = blk(x, attn_mask)

        # if self.upsample is not None:
        #     x = self.upsample(x)
        # if self.upsample is not None:
        #     x_down, Wh, Ww = self.upsample(x, H, W, padwh)
        #     # Wh, Ww = (H) * 2, (W) * 2
        #     return x_down, Wh, Ww
        # else:
        #     return x, H, W
        # return x

        return x, S, H, W



class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=True,  
                 use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, S, H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """

        # calculate attention mask for SW-MSA
        Sp = int(np.ceil(S / self.window_size)) * self.window_size
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Sp, Hp, Wp, 1), device=x.device)  
        s_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for s in s_slices:
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, s, h, w, :] = cnt
                    cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  
        mask_windows = mask_windows.view(-1,
                                         self.window_size * self.window_size * self.window_size) 
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
        if self.downsample is not None:
            x_down, padwh = self.downsample(x, S, H, W)
            Ws, Wh, Ww = (S + 1) // 2, (H + 1) // 2, (W + 1) // 2
            return x, S, H, W, x_down, Ws, Wh, Ww, padwh
        else:
            return x, S, H, W, x, S, H, W, [0,0,0]

class BasicLayer_up(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 upsample=True
                ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        
        # self.Upsample = upsample(dim=dim, norm_layer=norm_layer)
    def forward(self, x, S, H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
      
        x_up,_,_,_ = self.Upsample(x, S, H, W)
       
        # x_up+=skip
        S, H, W = S * 2, H * 2, W * 2
        # calculate attention mask for SW-MSA
        Sp = int(np.ceil(S / self.window_size)) * self.window_size
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Sp, Hp, Wp, 1), device=x.device) 
        s_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for s in s_slices:
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, s, h, w, :] = cnt
                    cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  
        mask_windows = mask_windows.view(-1,
                                         self.window_size * self.window_size * self.window_size)  
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        
        for blk in self.blocks:
            x_up = blk(x_up, attn_mask)
        
        return x_up, S, H, W
        
# done
class project(nn.Module):
    def __init__(self,in_dim,out_dim,stride,padding,activate,norm,last=False):
        super().__init__()
        self.out_dim=out_dim
        self.conv1=nn.Conv3d(in_dim,out_dim,kernel_size=3,stride=stride,padding=padding)
        self.conv2=nn.Conv3d(out_dim,out_dim,kernel_size=3,stride=1,padding=1)
        self.activate=activate()
        self.norm1=norm(out_dim)
        self.last=last  
        if not last:
            self.norm2=norm(out_dim)
            
    def forward(self,x):
        x=self.conv1(x)
        x=self.activate(x)
        #norm1
        Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm1(x)
        x = x.transpose(1, 2).view(-1, self.out_dim, Ws, Wh, Ww)
        

        x=self.conv2(x)
        if not self.last:
            x=self.activate(x)
            #norm2
            Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm2(x)
            x = x.transpose(1, 2).view(-1, self.out_dim, Ws, Wh, Ww)
        return x
        
    

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding

    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=4, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_3tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        img_size = [64,128,128]
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution

        self.proj1 = project(in_chans,embed_dim//2,[2,2,2],1,nn.GELU,nn.LayerNorm,False)
        self.proj2 = project(embed_dim//2,embed_dim,[1,2,2],1,nn.GELU,nn.LayerNorm,True)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, S, H, W = x.size()
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if S % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - S % self.patch_size[0]))
     
        x = self.proj1(x)  
        
        x = self.proj2(x)  
 
        if self.norm is not None:
            Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Ws, Wh, Ww)

        return x



class SwinTransformer(nn.Module):
    """ Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        pretrain_img_size (int): Input image size for training the pretrained model,
            used in absolute postion embedding. Default 224.
        patch_size (int | tuple(int)): Patch size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 pretrain_img_size=224,
                 patch_size=4,
                 in_chans=1  ,
                 embed_dim=96,
                 depths=[2, 2, 2, 2],
                 num_heads=[4, 8, 16, 32],
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_checkpoint=False):
        super().__init__()

        self.pretrain_img_size = pretrain_img_size

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            pretrain_img_size = to_3tuple(pretrain_img_size)
            patch_size = to_3tuple(patch_size)
            # �м���patch
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1],
                                  pretrain_img_size[2] // patch_size[2]]

            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1], patches_resolution[2]))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(
                    pretrain_img_size[0] // patch_size[0] // 2 ** i_layer, pretrain_img_size[1] // patch_size[1] // 2 ** i_layer,
                    pretrain_img_size[2] // patch_size[2] // 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(
                    depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging,
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    

    def forward(self, x):
        """Forward function."""
        
        x = self.patch_embed(x)
        # down=[]
       
        Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Ws, Wh, Ww), align_corners=True,
                                               mode='trilinear')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2) 
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        x_downsample = []
        x_downsample_resolutions = []
        padswh = []
        
      
        for i in range(self.num_layers):
            x_downsample.append(x)
            x_downsample_resolutions.append((Ws, Wh, Ww))
            layer = self.layers[i]
            x_out, S, H, W, x, Ws, Wh, Ww, padwh = layer(x, Ws, Wh, Ww)
            padswh.append(padwh)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)

                out = x_out.view(-1, S, H, W, self.num_features[i]).permute(0, 4, 1, 2, 3).contiguous()
              
                # down.append(out)
        return out, x_downsample, x_downsample_resolutions, S, H, W, padswh

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer, self).train(mode)
        self._freeze_stages()

class encoder(nn.Module):
    def __init__(self,
                 pretrain_img_size,
                 embed_dim,
                 patch_size=4,
                 depths=[2,2,2],
                 num_heads=[24,12,6],
                 window_size=4,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm, final_upsample="expand_first",
                 residual_patch_expand=True, patches_resolution=None
                 ):
        super().__init__()
        

        self.num_layers = len(depths)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.patches_resolution = patches_resolution

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        self.layers_cross_attention_up = nn.ModuleList()
        self.layers_patch_expand = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i_layer in range(self.num_layers):

            concat_linear = nn.Linear(2*int(embed_dim*2**(self.num_layers-1-i_layer)),
            int(embed_dim*2**(self.num_layers-1-i_layer))) if i_layer > 0 else nn.Identity()
            
            layer_up = BasicLayer_up(dim=int(embed_dim * 2 ** (self.num_layers-1-i_layer)),
                                     input_resolution=(patches_resolution[0] // (2 ** (self.num_layers-1-i_layer)),
                                                       patches_resolution[1] // (2 ** (self.num_layers-1-i_layer))),
                                     depth=depths[(self.num_layers-1-i_layer)],
                                     num_heads=num_heads[(self.num_layers-1-i_layer)],
                                     window_size=window_size,
                                     mlp_ratio=mlp_ratio,
                                     qkv_bias=qkv_bias, qk_scale=qk_scale,
                                     drop=drop_rate, attn_drop=attn_drop_rate,
                                     drop_path=dpr[sum(depths[:(self.num_layers-1-i_layer)]):sum(depths[:(self.num_layers-1-i_layer) + 1])],
                                     norm_layer=norm_layer,
                                     upsample=None)
            layer_cross_attention_up = BasicLayer_up_Xattn(dim=int(embed_dim * 2 ** (self.num_layers-1-i_layer + 1)),
                                                           input_resolution=(patches_resolution[0] // (2 ** (self.num_layers-1-i_layer)),
                                                                             patches_resolution[1] // (2 ** (self.num_layers-1-i_layer))),
                                                           skip_connection_resolution=(patches_resolution[0] // (2 ** (self.num_layers-1-i_layer)),
                                                                                       patches_resolution[1] // (2 ** (self.num_layers-1-i_layer))),
                                                           depth=1, # Only the cross attention
                                                           num_heads=num_heads[(self.num_layers-1-i_layer)],
                                                           window_size=window_size,
                                                           mlp_ratio=mlp_ratio,
                                                           qkv_bias=qkv_bias,
                                                           qk_scale=qk_scale,
                                                           drop=drop_rate,
                                                           attn_drop=attn_drop_rate,
                                                           drop_path=dpr[sum(depths[:(self.num_layers-1-i_layer)]):sum(depths[:(self.num_layers-1-i_layer) + 1])],
                                                           norm_layer=norm_layer,
                                                           residual_patch_expand=residual_patch_expand)
            # patch_expand = Patch_Expanding(input_resolution=(patches_resolution[0] // (2 ** (self.num_layers-1-i_layer)),
            #                                              patches_resolution[1] // (2 ** (self.num_layers-1-i_layer))),
            patch_expand = Patch_Expanding(dim=int(embed_dim * 2 ** (self.num_layers-1-i_layer + 1)))
            # patch_expand = Patch_Expanding(dim=int(embed_dim * 2 ** (i_layer + 1)))
                                       # dim_scale=2)
            self.layers.append(layer)
            self.concat_back_dim.append(concat_linear)
            self.layers_cross_attention_up.append(layer_cross_attention_up)
            self.layers_patch_expand.append(patch_expand)

        self.num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
    def forward(self,x, x_downsample, x_downsample_resolutions, Ws, Wh, Ww, padswh):

            
        outs=[]
        # # S, H, W = x.size(2), x.size(3), x.size(4)
        print('\n---->xb', x.shape,'\n')

        x = x.flatten(2).transpose(1, 2)
        # # for index,i in enumerate(skips): ##### à vérif ici
        # #      i = i.flatten(2).transpose(1, 2)
        # #      skips[index]=i
        # x = self.pos_drop(x)
            
        # for i in range(self.num_layers)[::-1]:
        #     # padwh = padswh[-(inx+2)]
        #     padwh = padswh[i+1]

        #     Ws_d, Wh_d, Ww_d = x_downsample_resolutions[2-i]
        #     skip_co = x_downsample[2-i]
        #     upsampling_blk = self.layers_cross_attention_up[i]
        #     upsampling_blk.input_resolution = (Wh, Ww)
        #     upsampling_blk.skip_connection_resolution = (Wh_d, Ww_d)

        #     x, Ws, Wh, Ww = self.layers_patch_expand[i](x, Ws, Wh, Ww, padwh)


        #     layer = self.layers[i]
        #     x, S, H, W,  = layer(x,skips[i], S, H, W)
        #     out = x.view(-1, S, H, W, self.num_features[i])
        #     outs.append(out)

        for inx, layer_up in enumerate(self.layers):
            padwh = padswh[-(inx+2)]
            
            Ws_d, Wh_d, Ww_d = x_downsample_resolutions[2-inx]
            skip_co = x_downsample[2-inx]
            upsampling_blk = self.layers_cross_attention_up[inx]
            upsampling_blk.input_resolution = (Ws, Wh, Ww)
            upsampling_blk.skip_connection_resolution = (Ws_d, Wh_d, Ww_d)
            
            x, Ws, Wh, Ww = self.layers_patch_expand[inx](x, Ws, Wh, Ww, padwh)

            x = torch.cat([x, skip_co],-1)
            x = self.concat_back_dim[inx](x)
            print("\n########## check")
            print('x', x.shape)
            print('Ws, Wh, Ww', Ws, Wh, Ww)
            x, Ws, Wh, Ww = layer_up(x, Ws, Wh, Ww)
            outs.append(x)

        # x = self.norm_up(x)  # B L C
  
        return outs
        # return outs










        
class final_patch_expanding(nn.Module):
    def __init__(self,dim,num_class,patch_size):
        super().__init__()
        self.up=nn.ConvTranspose3d(dim,num_class,patch_size,patch_size)

    def forward(self,x):
        x=x.permute(0,4,1,2,3)
        x=self.up(x)
       
        return x    




                                         
class swintransformer(SegmentationNetwork):

    def __init__(self, input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage=2,
                 feat_map_mul_on_downscale=2, conv_op=nn.Conv2d,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, deep_supervision=True, dropout_in_localization=False,
                 final_nonlin=softmax_helper, weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None,
                 conv_kernel_sizes=None,
                 upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False,
                 max_num_features=None, basic_block=None,
                 seg_output_use_bias=False):
    
        super(swintransformer, self).__init__()
        
        
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision
        self.num_classes=num_classes
        self.conv_op=conv_op
       
        
        self.upscale_logits_ops = []
     
        
        self.upscale_logits_ops.append(lambda x: x)
        
        
        embed_dim=192
        depths=[2, 2, 2, 2]
        num_heads=[6, 12, 24, 48]
        patch_size=[2,4,4]
        self.model_down=SwinTransformer(pretrain_img_size=[64,128,128],window_size=4,embed_dim=embed_dim,patch_size=patch_size,depths=depths,num_heads=num_heads,in_chans=1)
        self.encoder=encoder(pretrain_img_size=[64,128,128],embed_dim=embed_dim,window_size=4,patch_size=patch_size,num_heads=[24,12,6],depths=[2,2,2], final_upsample="expand_first", residual_patch_expand=True, patches_resolution=self.model_down.patches_resolution)
   
        self.final=[]
        self.final.append(final_patch_expanding(embed_dim*2**0,14,patch_size=(2,4,4)))
        for i in range(1,len(depths)-1):
            self.final.append(final_patch_expanding(embed_dim*2**i,14,patch_size=(4,4,4)))
        self.final=nn.ModuleList(self.final)
        
    def forward(self, x):
        
            
            
        seg_outputs=[]
        # skips = self.model_down(x)
        neck, x_downsample, x_downsample_resolutions, Ws, Wh, Ww, padswh = self.model_down(x)
        # neck=skips[-1]
       
        # out=self.encoder(neck,skips)
        out = self.encoder(neck, x_downsample, x_downsample_resolutions, Ws, Wh, Ww, padswh)
        
        for i in range(len(out)):  
            seg_outputs.append(self.final[-(i+1)](out[i]))

        print("Je pari qu'il est 2h du mat (23h50 quand j'écris ça). Franchement va dormir !!! (en vrai bravo)")
        exit(0)

        if self._deep_supervision and self.do_ds:
            return tuple([seg_outputs[-1]] + [i(j) for i, j in
                                              zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
        else:
            
            return seg_outputs[-1]
        
        
        
   

    @staticmethod
    def compute_approx_vram_consumption(patch_size, num_pool_per_axis, base_num_features, max_num_features,
                                        num_modalities, num_classes, pool_op_kernel_sizes, deep_supervision=False,
                                        conv_per_stage=2):
        """
        This only applies for num_conv_per_stage and convolutional_upsampling=True
        not real vram consumption. just a constant term to which the vram consumption will be approx proportional
        (+ offset for parameter storage)
        :param deep_supervision:
        :param patch_size:
        :param num_pool_per_axis:
        :param base_num_features:
        :param max_num_features:
        :param num_modalities:
        :param num_classes:
        :param pool_op_kernel_sizes:
        :return:
        """
        if not isinstance(num_pool_per_axis, np.ndarray):
            num_pool_per_axis = np.array(num_pool_per_axis)

        npool = len(pool_op_kernel_sizes)

        map_size = np.array(patch_size)
        tmp = np.int64((conv_per_stage * 2 + 1) * np.prod(map_size, dtype=np.int64) * base_num_features +
                       num_modalities * np.prod(map_size, dtype=np.int64) +
                       num_classes * np.prod(map_size, dtype=np.int64))

        num_feat = base_num_features

        for p in range(npool):
            for pi in range(len(num_pool_per_axis)):
                map_size[pi] /= pool_op_kernel_sizes[p][pi]
            num_feat = min(num_feat * 2, max_num_features)
            num_blocks = (conv_per_stage * 2 + 1) if p < (npool - 1) else conv_per_stage  # conv_per_stage + conv_per_stage for the convs of encode/decode and 1 for transposed conv
            tmp += num_blocks * np.prod(map_size, dtype=np.int64) * num_feat
            if deep_supervision and p < (npool - 2):
                tmp += np.prod(map_size, dtype=np.int64) * num_classes
            # print(p, map_size, num_feat, tmp)
        return tmp
