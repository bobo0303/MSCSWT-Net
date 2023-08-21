import logging
import numpy as np
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange

logger = logging.getLogger(__name__)


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu(x)


class GELU2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class CSwinAttention_block(nn.Module):
    def __init__(self, args, dim, split_size=4, channel=256, dim_out=None, num_heads=2, attn_drop=0., proj_drop=0.,
                 qk_scale=None):
        super().__init__()

        self.dim = dim
        self.dim_out = dim_out or dim
        self.num_heads = num_heads
        self.attn_drop = nn.Dropout(attn_drop)
        self.cswinln = nn.LayerNorm(channel, eps=1e-4)
        self.cswinln_h = nn.LayerNorm(int(channel / 2), eps=1e-4)
        self.cswinln_w = nn.LayerNorm(int(channel / 2), eps=1e-4)

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.sp = split_size

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_qkv_h = nn.Linear(int(dim / 2), int(dim * 3 / 2), bias=False)
        self.to_qkv_w = nn.Linear(int(dim / 2), int(dim * 3 / 2), bias=False)

        # lepe conv from set_lepe_conv

        # self.proj = nn.LazyConv2d(dim, kernel_size=1)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    def shift_featuremap(self, x):
        feature_size = x.size(2)
        splic_c = int(x.size(1) / 2)
        x = torch.split(x, [splic_c, splic_c], dim=1)
        feature_list = [1] * feature_size
        x_h = torch.split(x[0], feature_list, dim=3)
        x_w = torch.split(x[1], feature_list, dim=2)
        xn_shifted = []
        xm_shifted = []
        current_shift_step = 0
        # H W
        for n, m in zip(x_h, x_w):
            rolled_n = torch.roll(n, current_shift_step, dims=2)
            rolled_m = torch.roll(m, current_shift_step, dims=3)
            xn_shifted.append(rolled_n)
            xm_shifted.append(rolled_m)
            current_shift_step += 1
        xh_shifted = torch.cat(xn_shifted, dim=2)
        xw_shifted = torch.cat(xm_shifted, dim=3
        return xh_shifted, xw_shifted

    def restore_featuremap(self, x):

        feature_size = x.size(2)
        splic_c = int(x.size(1) / 2)
        x = torch.split(x, [splic_c, splic_c], dim=1)
        feature_list = [1] * feature_size
        x_h = torch.split(x[0], feature_list, dim=3)
        x_w = torch.split(x[1], feature_list, dim=2)

        xh_restored = []
        xw_restored = []
        current_shift_step = 0
        for n, m in zip(x_h, x_w):
            rolled_n = torch.roll(n, -current_shift_step, dims=2)
            rolled_m = torch.roll(m, -current_shift_step, dims=3)
            xh_restored.append(rolled_n)
            xw_restored.append(rolled_m)
            current_shift_step += 1
        xh_restored = torch.cat(xh_restored, dim=3)
        xw_restored = torch.cat(xw_restored, dim=2)
        return torch.cat((xh_restored, xw_restored), dim=1)

    def set_lepe_conv(self, x: torch.Tensor) -> nn.Conv2d:
        """input_size : [B, C, H', W']"""
        dim = x.size(1)
        # init lepe conv
        # self.lepe_conv = nn.LazyConv2d(dim, kernel_size=3, stride=1, padding=1, groups=dim).to(x.device)
        self.lepe_conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim).to(x.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """input_size : [B, C, H, W]  or [B, L, C]"""
        # create a list for later concat
        attened_x = []
        attened_att = []

        if len(x.shape) == 3:  # [B, L, C]
            B, L, C = x.shape
            H = W = int(np.sqrt(L))

        elif len(x.shape) == 4:  # [B, C, H, W]
            B, C, H, W = x.shape

        assert H % self.sp == 0 and W % self.sp == 0, \
            f'H={H} or W={W} cannot be divided by split_size={self.sp} '

        condition = (H == self.sp and W == self.sp)  # feature size == split size, one attn operation
        # feature size  > split size, two attn operations

        if condition:
            h = w = 1
            hsp = wsp = self.sp  # full feature
            param = [(h, hsp, w, wsp)]

        else:
            h1, hsp_1, w_1, wsp_1 = H // self.sp, self.sp, 1, W  # vertical window and horizontal shift
            h2, hsp_2, w_2, wsp_2 = 1, H, W // self.sp, self.sp  # horizontal window and vertical shift
            param = [(h1, hsp_1, w_1, wsp_1), (h2, hsp_2, w_2, wsp_2)]

        if condition:
            x_patch = rearrange(x, 'b c h w -> b (h w) c')
            x_patch = self.cswinln(x_patch)
            qkv = self.to_qkv(x_patch).chunk(3, dim=-1)
            qkv = [qkv]

        else:
            x_h, x_w = self.shift_featuremap(x)
            x_patch_h = rearrange(x_h, 'b c h w -> b (h w) c')
            x_patch_w = rearrange(x_w, 'b c h w -> b (h w) c')
            x_patch_h = self.cswinln_h(x_patch_h)
            x_patch_w = self.cswinln_w(x_patch_w)
            qkv_h = self.to_qkv_h(x_patch_h).chunk(3, dim=-1)
            qkv_w = self.to_qkv_w(x_patch_w).chunk(3, dim=-1)
            (q1), (k1), (v1) = qkv_h
            (q2), (k2), (v2) = qkv_w
            qkv = [(q1, k1, v1), (q2, k2, v2)]

        for index, (x, (h, hsp, w, wsp)) in enumerate(zip(qkv, param)):
            # print(h, hsp, w, wsp)
            # cswin format
            q, k, v = map(lambda t: rearrange(t, 'b (h hsp w wsp) (c head)  -> (b h w) head (hsp wsp) c',
                                              head=self.num_heads, h=h, w=w, hsp=hsp, wsp=wsp), x)

            # from [(B * H/hsp * W/wsp), head, (hsp * wsp), C/head] to [(B * H/hsp * W/wsp), C, hsp, wsp]
            lepe = rearrange(v, '(b h w) head (hsp wsp) c -> (b h w) (c head) hsp wsp',
                             head=self.num_heads, h=h, w=w, hsp=hsp, wsp=wsp)

            # set lepe_conv
            self.set_lepe_conv(lepe)  ###

            # lepe_conv
            lepe = self.lepe_conv(lepe)

            # back to [(B * H/hsp * W/wsp), head, (hsp * wsp), C/head]
            lepe = rearrange(lepe, '(b h w) (c head) hsp wsp -> (b h w) head (hsp wsp) c',
                             head=self.num_heads, h=h, w=w, hsp=hsp, wsp=wsp)
            # print(f'{lepe.shape=}')

            # attention
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N
            attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
            attn = self.attn_drop(attn)

            x = (attn @ v) + lepe

            # [(B * H / hsp * W / wsp), head, (hsp * wsp), C / head] to[(B , C, H, W]
            x = rearrange(x, '(b h w) head (hsp wsp) c -> b (c head) (h hsp) (w wsp)',
                          head=self.num_heads, h=h, w=w, hsp=hsp, wsp=wsp)

            attened_x.append(x)
            attened_att.append(attn)

        x = self.proj(torch.cat(attened_x, dim=1))
        x = self.restore_featuremap(x) if hsp != wsp else x

        return x, attened_att


class shifted_cswin_Transformer(CSwinAttention_block):

    def __init__(self, args, layer=None):
        super().__init__(args, dim=256, split_size=4, dim_out=None, num_heads=2, attn_drop=0., proj_drop=0.,
                         qk_scale=None)

        self.n_embd = args['n_embd']
        self.num_heads = args['head']
        self.dim = args['dim']
        self.split_size = args['split_size']
        self.layer = layer

        if self.layer == 'first':
            self.CSwinAttention_block = CSwinAttention_block(args, self.dim[0], split_size=self.split_size[0],
                                                             channel=self.n_embd[0],
                                                             dim_out=None, num_heads=self.num_heads[0], attn_drop=0.,
                                                             proj_drop=0., qk_scale=None)
            self.ln_mlp = nn.LayerNorm(self.n_embd[0])
            self.mlp = nn.Sequential(
                nn.Linear(self.n_embd[0], 4 * self.n_embd[0]),
                GELU2(),  # sigmoid
                nn.Linear(4 * self.n_embd[0], self.n_embd[0]),
                nn.Dropout(args['resid_pdrop']),
            )
            self.loop_time = args['loop_time'][0]

        elif self.layer == 'second':
            self.CSwinAttention_block = CSwinAttention_block(args, self.dim[1], split_size=self.split_size[1],
                                                             channel=self.n_embd[1],
                                                             dim_out=None, num_heads=self.num_heads[1], attn_drop=0.,
                                                             proj_drop=0., qk_scale=None)
            self.ln_mlp = nn.LayerNorm(self.n_embd[1])
            self.mlp = nn.Sequential(
                nn.Linear(self.n_embd[1], 4 * self.n_embd[1]),
                GELU2(),  # sigmoid
                nn.Linear(4 * self.n_embd[1], self.n_embd[1]),
                nn.Dropout(args['resid_pdrop']),
            )
            self.loop_time = args['loop_time'][1]

        elif self.layer == 'third':
            self.CSwinAttention_block = CSwinAttention_block(args, self.dim[2], split_size=self.split_size[2],
                                                             channel=self.n_embd[2],
                                                             dim_out=None, num_heads=self.num_heads[2], attn_drop=0.,
                                                             proj_drop=0., qk_scale=None)
            self.ln_mlp = nn.LayerNorm(self.n_embd[2])
            self.mlp = nn.Sequential(
                nn.Linear(self.n_embd[2], 4 * self.n_embd[2]),
                GELU2(),  # sigmoid
                nn.Linear(4 * self.n_embd[2], self.n_embd[2]),
                nn.Dropout(args['resid_pdrop']),
            )
            self.loop_time = args['loop_time'][2]

        elif self.layer == 'fourth':
            self.CSwinAttention_block = CSwinAttention_block(args, self.dim[3], split_size=self.split_size[3],
                                                             channel=self.n_embd[3],
                                                             dim_out=None, num_heads=self.num_heads[3], attn_drop=0.,
                                                             proj_drop=0., qk_scale=None)
            self.ln_mlp = nn.LayerNorm(self.n_embd[3])
            self.mlp = nn.Sequential(
                nn.Linear(self.n_embd[3], 4 * self.n_embd[3]),
                GELU2(),  # sigmoid
                nn.Linear(4 * self.n_embd[3], self.n_embd[3]),
                nn.Dropout(args['resid_pdrop']),
            )
            self.loop_time = args['loop_time'][3]

    def forward(self, x):
        [b, c, h, w] = x.shape

        xo = x.clone()
        for _ in range(self.loop_time):
            x_cs_att, _ = self.CSwinAttention_block(x)
            x = xo + x_cs_att
            x_r = rearrange(x, 'b c h w -> b (h w) c')
            x = x + self.mlp(self.ln_mlp(x_r)).reshape(b, h, w, c).permute(0, 3, 1, 2)
            x = x.reshape(b, h, w, c).permute(0, 3, 1, 2)

        return x
