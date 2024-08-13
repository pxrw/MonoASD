import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange, repeat

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass


class SelectScan2D(nn.Module):
    def __init__(self, d_model, d_state=8, d_conv=3, expand=1, dt_rank='auto', dt_min=0.001, dt_max=0.1,
                 dt_init='random', dt_scale=1.0, dt_init_floor=1e-4, dropout=0., conv_bias=True,
                 bias=False, device=None, dtype=None, **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}
        super(SelectScan2D, self).__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == 'auto' else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init='random', dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == 'constant':
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)

        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)

        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, 'd n -> r d n', r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        K = 4

        # [b, c, h, w] -> [b, c, w, h] -> [b, c, l] stack [b, c, l] -> [b, 2c, l] -> [b, 2, c, l]
        x_hwwh = torch.stack([x.view(B, -1, L), x.transpose(2, 3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x, H, W, relative_pos=None):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, H, W, C)
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # chunk()方法能按照维度，对张量进行均匀切分，返回结果是院张量的视图

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = y.transpose(1, 2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)

        if self.dropout is not None:
            out = self.dropout(out)
        out = out.reshape(B, N, C)
        return out


class PatchMerging2D_sentence(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super(PatchMerging2D_sentence, self).__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, int(math.sqrt(N)), int(math.sqrt(N)), C)
        B, H, W, C = x.shape
        # print(f'x shape: {x.shape}')

        SHAPE_FIX = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):
            print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2

        # 每隔一个像素进行采样
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C

        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]

        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, H // 2, W // 2, 4 * C)

        x = self.norm(x)
        x = self.reduction(x)  # c*4 -> c*2
        b, h, w, c = x.shape
        x = x.reshape(b, h * w, c)  # [b, n', c']  n' = h/2 * w/2, c' = c*2
        return x, h, w


class PatchMerging2D_word(nn.Module):
    def __init__(self, dim_in, dim_out, stride=2, act_layer=nn.GELU):
        super(PatchMerging2D_word, self).__init__()
        self.stride = stride
        self.dim_out = dim_out
        self.norm = nn.LayerNorm(dim_in)
        self.conv = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=2 * stride - 1, padding=stride - 1, stride=stride)
        )

    def forward(self, x, H_out, W_out, H_in, W_in):
        B_N, M, C = x.shape
        x = self.norm(x)
        x = x.reshape(-1, H_out, W_out, H_in, W_in, C)
        pad_input = (H_out % 2 == 1) or (W_out % 2 == 1)
        if pad_input:
            x = F.pad(x.permute(0, 3, 4, 5, 1, 2), (0, W_out % 2, 0, H_out % 2))
            x = x.permute(0, 4, 5, 1, 2, 3)

        H, W = x.shape[1], x.shape[2]
        SHAPE_FIX = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):
            print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C

        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]

        x = torch.cat([torch.cat([x0, x1], 3), torch.cat([x2, x3], 3)], 4)  # B, H/2, W/2, 2*H_in, 2*W_in, C
        x = x.reshape(-1, 2 * H_in, 2 * W_in, C).permute(0, 3, 1, 2)  # B_N/4, C, 2*H_in, 2*W_in
        x = self.conv(x)  # B_N/4, C, H_in, W_in
        x = x.reshape(-1, self.dim_out, M).transpose(1, 2)
        return x


class Block(nn.Module):
    def __init__(self, outer_dim, inner_dim, num_words, drop_path=0., norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.has_inner = inner_dim > 0
        if self.has_inner:
            # print(f'inner dim: {inner_dim}, num words: {num_words}')
            self.inner_norm1 = norm_layer(num_words * inner_dim)
            self.inner_attn = SelectScan2D(inner_dim, dropout=0, d_state=16)
            self.proj_norm1 = norm_layer(num_words * inner_dim)
            self.proj = nn.Linear(num_words * inner_dim, outer_dim, bias=False)
            self.proj_norm2 = norm_layer(outer_dim)

        self.outer_norm1 = norm_layer(outer_dim)
        self.outer_attn = SelectScan2D(d_model=outer_dim, dropout=0, d_state=16)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, outer_tokens, H_out, W_out, H_in, W_in, relative_pos):
        B, N, C = outer_tokens.size()
        # print(f'in Block, x shape: {x.shape}, outer_tokens shape: {outer_tokens.shape}')
        if self.has_inner:
            x = x + self.drop_path(
                self.inner_attn(self.inner_norm1(x.reshape(B, N, -1)).reshape(B * N, H_in * W_in, -1), H_in,
                                W_in))  # B*N, k*k, c

            outer_tokens = outer_tokens + self.proj_norm2(self.proj(self.proj_norm1(x.reshape(B, N, -1))))
        outer_tokens = outer_tokens + self.drop_path(
            self.outer_attn(self.outer_norm1(outer_tokens), H_out, W_out, relative_pos))
        return x, outer_tokens


class Stage(nn.Module):
    def __init__(self, num_blocks, outer_dim, inner_dim, outer_head, num_patches, num_words, drop_path=0.,
                 norm_layer=nn.LayerNorm, sr_ratio=1):
        super(Stage, self).__init__()
        blocks = []
        drop_path = drop_path if isinstance(drop_path, list) else [drop_path] * num_blocks

        for j in range(num_blocks):
            if j == 0:
                _inner_dim = inner_dim
            elif j == 1 and num_blocks > 6:
                _inner_dim = inner_dim
            else:
                _inner_dim = -1
            blocks.append(Block(
                outer_dim, _inner_dim, num_words=num_words, drop_path=drop_path[j], norm_layer=norm_layer,
            ))
        self.blocks = nn.ModuleList(blocks)
        self.relative_pos = nn.Parameter(torch.randn(
            1, outer_head, num_patches, num_patches // sr_ratio // sr_ratio
        ))

    def forward(self, inner_tokens, outer_tokens, H_out, W_out, H_in, W_in):
        for blk in self.blocks:
            inner_tokens, outer_tokens = blk(inner_tokens, outer_tokens, H_out, W_out, H_in, W_in, self.relative_pos)
        return inner_tokens, outer_tokens


class CBR(nn.Module):
    def __init__(self, in_channs, out_channs, kernel_size=3, stride=1, padding=1, inplace=False):
        super(CBR, self).__init__()
        self.sq = nn.Sequential(
            nn.Conv2d(in_channs, out_channs, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channs),
            nn.ReLU(inplace=inplace)
        )

    def forward(self, x):
        return self.sq(x)


class ConvBlock(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale=1e-6, kernel_size=3):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(dim, eps=1e-5)
        self.act1 = nn.GELU(approximate='tanh')

        self.conv2 = nn.Conv2d(dim, dim, kernel_size, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(dim, eps=1e-5)
        self.layer_scale = layer_scale

        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.gamma = nn.Parameter(layer_scale * torch.ones(dim))
            self.layer_scale = True
        else:
            self.layer_scale = False
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        if self.layer_scale:
            x = x * self.gamma.view(1, -1, 1, 1)
        x = input + self.drop_path(x)
        return x


class Stem(nn.Module):
    '''
    输入数据应满足[B, C, H, W]维度
    return:
        inner tokens: [B*(H/8)*(W/8), 16, inner_dim]
        outer tokens: [B, (H/8)*(W/8), -1]
    '''

    def __init__(self, img_size, in_channs=3, outer_dim=32, inner_dim=8):
        super(Stem, self).__init__()
        # img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.inner_dim = inner_dim
        self.num_patches = (img_size[0] // 8) * (img_size[1] // 8)
        self.num_words = 16

        self.common_convs = CBR(in_channs, inner_dim * 2, kernel_size=3, stride=2, padding=1, inplace=False)
        self.inner_convs = CBR(inner_dim * 2, inner_dim, kernel_size=3, stride=1, padding=1, inplace=False)
        self.outer_convs = nn.Sequential(
            CBR(inner_dim * 2, inner_dim * 4, kernel_size=3, stride=2, padding=1, inplace=True),
            CBR(inner_dim * 4, inner_dim * 8, kernel_size=3, stride=2, padding=1, inplace=True),
            CBR(inner_dim * 8, outer_dim, kernel_size=3, stride=1, padding=1, inplace=False)
        )
        # nn.Unfold(kernel_size, dilation, padding, stride)，一个批次的输入样本中，提取出滑动的局部区域块
        # input(B, C, H, W)
        # return(B, N, L), N:表示生成后每个局部块的大小 N = C * kernel_h * kernel_w, L = (H / kernel_h) * (W / kernel_w)
        self.unfold = nn.Unfold(kernel_size=4, padding=0, stride=4)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.common_convs(x)  # [B, C, H, W] -> [B, 2C', H/2, W/2]   C'=inner_dim
        # 每个'sentence'对应图像中4*4的区域
        H_out, W_out = H // 8, W // 8
        # 每个'sentence'由4*4个'word'组成，每个'word'对应图像中1*1的区域
        H_in, W_in = 4, 4

        # inner_tokens
        inner_tokens = self.inner_convs(x)  # [B, 2C', H/2, W/2] -> [B, C', H/2, W/2]
        # [B, C', H/2, W/2] -> [B, C'*16, (H/8)*(W/8)] -> [B, L, C'*16]
        inner_tokens = self.unfold(inner_tokens).transpose(1, 2)

        # print(f'in stem, inner dim: {self.inner_dim}')
        # [B, (H/8)*(W/8), C'*16] -> [B*(H/8)*(W/8), C', 16] -> [B*N, 16, C']
        inner_tokens = inner_tokens.reshape(B * H_out * W_out, self.inner_dim, H_in * W_in).transpose(1, 2)

        # outer_tokens
        # [B, C, H/2, W/2] -> [B, 768, H/8, W/8] -> [B, H/8, W/8, 768] -> [B, (H/8)*(W/8), -1]
        outer_tokens = self.outer_convs(x)
        outer_tokens = outer_tokens.permute(0, 2, 3, 1).reshape(B, H_out * W_out, -1)
        # print(f'inner tokens: {inner_tokens.shape} \nouter tokens: {outer_tokens.shape}')
        return inner_tokens, outer_tokens, (H_out, W_out), (H_in, W_in)


class UpSampleBlock(nn.Module):
    def __init__(self, in_channs, out_channs):
        super(UpSampleBlock, self).__init__()
        self.transpose_conv = nn.ConvTranspose2d(
            in_channs, out_channs, kernel_size=2, stride=2, padding=0
        )
        self.batch_norm1 = nn.BatchNorm2d(out_channs)
        # GeLU 激活函数
        self.gelu1 = nn.GELU()
        # 步长为1的3x3卷积
        self.conv = nn.Conv2d(
            out_channs, out_channs, kernel_size=3, stride=1, padding=1
        )
        # 另一个批量归一化
        self.batch_norm2 = nn.BatchNorm2d(out_channs)
        # 另一个 GeLU 激活函数
        self.gelu2 = nn.GELU()

    def forward(self, x):
        x = self.transpose_conv(x)
        x = self.conv(self.gelu1(self.batch_norm1(x)))
        x = self.batch_norm2(x)
        return self.gelu2(x)


class MambaEncoder(nn.Module):
    def __init__(self, feat_size, in_channs=3, inner_dim=24, out_dims=64):
        super(MambaEncoder, self).__init__()
        self.patch_embed = Stem(
            img_size=feat_size, in_channs=in_channs, outer_dim=out_dims, inner_dim=inner_dim
        )

        num_patches = self.patch_embed.num_patches
        num_words = self.patch_embed.num_words
        # print(f'num patches: {num_patches}, num words: {num_words}')

        self.stage = Stage(
            num_blocks=1, outer_dim=out_dims, inner_dim=inner_dim, outer_head=4, num_patches=num_patches,
            num_words=num_words, drop_path=0
        )

        self.up_block = UpSampleBlock(out_dims, out_dims)

    def forward(self, x):
        inner_tokens, outer_tokens, (H_out, W_out), (H_in, W_in) = self.patch_embed(x)
        # print(f'inner tokens: {inner_tokens.shape} \nouter tokens: {outer_tokens.shape}')
        inner_tokens, outer_tokens = self.stage(inner_tokens, outer_tokens, H_out, W_out, H_in, W_in)
        # print(f'inner tokens shape:{inner_tokens.shape}, outer tokens shape:{outer_tokens.shape}')
        # B, L, M = outer_tokens.shape

        # mid_out = outer_tokens.reshape(B, int(math.sqrt(L)), int(math.sqrt(L)), M).permute(0, 3, 1, 2)
        # print(f'mid out shape: {mid_out.shape}')
        # mid_out = self.up_block(mid_out)
        return outer_tokens


class ConvMamba(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=4,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            conv_bias=True,
            bias=False,
            use_fast_path=True,
            layer_idx=None,
            device=None,
            dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.x_proj = nn.Linear(
            self.d_inner // 2, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner // 2, bias=True, **factory_kwargs)
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(self.d_inner // 2, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner // 2,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner // 2, device=device))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.conv1d_x = nn.Conv1d(
            in_channels=self.d_inner // 2,
            out_channels=self.d_inner // 2,
            bias=conv_bias // 2,
            kernel_size=d_conv,
            groups=self.d_inner // 2,
            **factory_kwargs,
        )
        self.conv1d_z = nn.Conv1d(
            in_channels=self.d_inner // 2,
            out_channels=self.d_inner // 2,
            bias=conv_bias // 2,
            kernel_size=d_conv,
            groups=self.d_inner // 2,
            **factory_kwargs,
        )

    def forward(self, hidden_states):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        _, seqlen, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)
        xz = rearrange(xz, "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)
        A = -torch.exp(self.A_log.float())

        x = F.silu(F.conv1d(input=x, weight=self.conv1d_x.weight, bias=self.conv1d_x.bias, padding='same',
                            groups=self.d_inner // 2))
        z = F.silu(F.conv1d(input=z, weight=self.conv1d_z.weight, bias=self.conv1d_z.bias, padding='same',
                            groups=self.d_inner // 2))
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        y = selective_scan_fn(x, dt, A, B, C, self.D.float(), z=None,
                              delta_bias=self.dt_proj.bias.float(),
                              delta_softplus=True,
                              return_last_state=None)

        y = torch.cat([y, z], dim=1)
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        return out


if __name__ == '__main__':
    input = torch.randn(1, 256, 24, 80).cuda()
    input = input.flatten(start_dim=2)
    input = rearrange(input, 'b c n -> b n c')
    print(f'input shape: {input.shape}')
    # m = MambaEncoder(feat_size=(24, 80), in_channs=256, inner_dim=4, out_dims=16).cuda()
    # m = SelectScan2D(d_model=256).cuda()
    m = ConvMamba(d_model=256).cuda()
    out = m(input)
    print(f'out shape: {out.shape}')
