import torch
from torch import nn
from timm.models.layers import DropPath
from einops.layers.torch import Rearrange
import torch.nn.functional as F

# from lib.align.psp import PSP


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect', bias=True)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.concat([x_avg, x_max], dim=1)
        sattn = self.sa(x2)
        return sattn


class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=8):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )

    def forward(self, x):
        x_gap = self.gap(x)
        cattn = self.ca(x_gap)
        return cattn


class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2)  # B, C, 1, H, W
        pattn1 = pattn1.unsqueeze(dim=2)  # B, C, 1, H, W
        x2 = torch.cat([x, pattn1], dim=2)  # B, C, 2, H, W
        x2 = Rearrange('b c t h w -> b (c t) h w')(x2)
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2


class LargeKernelAttention(nn.Module):
    def __init__(self, dim):
        super(LargeKernelAttention, self).__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim // 2, dim, 1)

    def forward(self, x):
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)

        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)
        attn = self.conv(attn)
        return attn


# feature align module-------------------------------------------------------------------------------------------
class FeatureAlign(nn.Module):
    def __init__(self, dim):
        super(FeatureAlign, self).__init__()
        self.stu_attn = LargeKernelAttention(dim)
        self.tea_attn = LargeKernelAttention(dim)

    def forward(self, stu, tea):
        # print(f'cur stu shape:{stu.shape}, cur tea shape: {tea.shape}')
        attn_stu = self.stu_attn(stu)
        attn_tea = self.tea_attn(tea)

        stu = stu * attn_tea
        tea = tea * attn_stu
        return stu, tea


class FeatureAlignLoss(nn.Module):
    def __init__(self, dim_list):
        super(FeatureAlignLoss, self).__init__()
        if isinstance(dim_list, list):
            for i in range(len(dim_list)):
                feature_align = FeatureAlign(dim_list[i])
                self.__setattr__(f'feature_align_{i}', feature_align)

    def forward(self, s, t):
        align_loss = 0.0
        stu_feats, tea_feats = [], []
        if isinstance(s, list):
            for i in range(len(s)):
                m = self.__getattr__(f'feature_align_{i}')
                b, c, h, w = s[i].shape
                cur_stu = s[i]
                cur_tea = t[i]

                stu, tea = m(cur_stu, cur_tea)
                # stu_mean = torch.mean(stu, dim=1, keepdim=True)
                # tea_mean = torch.mean(tea, dim=1, keepdim=True)
                # align_loss += F.mse_loss(stu_mean, tea_mean, reduction='mean')
                stu = self.norm(stu)
                tea = self.norm(tea)
                # log_pred_stu = F.log_softmax(stu, dim=1)
                # pred_tea = F.softmax(tea, dim=1)
                align_loss += F.l1_loss(stu, tea, reduction='mean') / b

                stu = stu.reshape(b, c, h, w)
                tea = tea.reshape(b, c, h, w)

                cur_stu = cur_stu + stu
                cur_tea = cur_tea + tea

                stu_feats.append(cur_stu)
                tea_feats.append(cur_tea)

                # align_loss += F.kl_div(log_pred_stu, pred_tea, reduction='none').sum(1).mean()

        return align_loss, stu_feats, tea_feats

    def norm(self, logit):
        b, c, h, w = logit.shape
        logit = logit.reshape(b, c, -1)
        mean = logit.mean(dim=-1, keepdims=True)
        std = logit.std(dim=-1, keepdims=True)
        return (logit - mean) / (1e-7 + std)


if __name__ == '__main__':
    x1 = torch.randn(1, 64, 96, 320)
    x2 = torch.randn(1, 128, 48, 160)
    x3 = torch.randn(1, 256, 24, 80)
    x4 = torch.randn(1, 512, 12, 40)

    stu = [x1, x2, x3, x4]
    tea = [x1, x2, x3, x4]

    # m = FeatureAlign(dim=64)
    # out1, out2 = m(x1, x1)
    # print(out1.shape, out2.shape)

    m = FeatureAlignLoss([64, 128, 256, 512])
    loss, stu_feats, tea_feats = m(stu, tea)
    for item in stu_feats:
        print(item.shape)
    print('#########################')
    for item in tea_feats:
        print(item.shape)
