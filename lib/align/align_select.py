import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.align.fusion_align import FeatureAlign
from lib.align.mamba_select_align import ConvMamba

class CBR(nn.Module):
    def __init__(self, in_channs, out_channs, kernel_size=3, stride=1, padding=1):
        super(CBR, self).__init__()
        self.sq = nn.Sequential(
            nn.Conv2d(in_channs, out_channs, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channs),
            nn.GELU()
        )

    def forward(self, x):
        return self.sq(x)

class AlignSelectLoss(nn.Module):
    def __init__(self, dim_list):
        super(AlignSelectLoss, self).__init__()
        self.dim_list = dim_list

        for i in range(len(dim_list)):
            feat_align = FeatureAlign(dim_list[i])
            self.__setattr__(f'feat_align_{i}', feat_align)

            conv_ssm_stu = ConvMamba(dim_list[i])
            conv_ssm_tea = ConvMamba(dim_list[i])
            self.__setattr__(f'conv_ssm_stu_{i}', conv_ssm_stu)
            self.__setattr__(f'conv_ssm_tea_{i}', conv_ssm_tea)

            conv_stu = CBR(dim_list[i] * 3, dim_list[i])
            conv_tea = CBR(dim_list[i] * 3, dim_list[i])
            self.__setattr__(f'conv_stu_{i}', conv_stu)
            self.__setattr__(f'conv_tea_{i}', conv_tea)

    def forward(self, stu, tea):
        align_loss = 0.0
        select_loss = 0.0
        loss = 0.0
        out_stu, out_tea = [], []

        for i in range(len(self.dim_list)):
            cur_stu = stu[i]
            cur_tea = tea[i]
            b, c, h, w = cur_stu.shape

            feat_align = self.__getattr__(f'feat_align_{i}')
            aligned_stu, aligned_tea = feat_align(cur_stu, cur_tea)

            aligned_stu_norm, aligned_tea_norm = self.norm(aligned_stu.flatten(start_dim=2)), self.norm(aligned_tea.flatten(start_dim=2))
            align_loss += F.l1_loss(aligned_stu_norm, aligned_tea_norm, reduction='mean') / b

            # ------------------------------------------------------------------------
            cur_stu_flatten = cur_stu.flatten(start_dim=2).permute(0, 2, 1) # [B, C, H, W] -> [B, C, N] -> [B, N, C]
            cur_tea_flatten = cur_tea.flatten(start_dim=2).permute(0, 2, 1)

            conv_ssm_stu = self.__getattr__(f'conv_ssm_stu_{i}')
            conv_ssm_tea = self.__getattr__(f'conv_ssm_tea_{i}')
            selected_stu = conv_ssm_stu(cur_stu_flatten)
            selected_tea = conv_ssm_tea(cur_tea_flatten)

            selected_stu = selected_stu.permute(0, 2, 1)  # [B, N, C] -> [B, C, N]
            selected_tea = selected_tea.permute(0, 2, 1)

            selected_stu_norm = self.norm(selected_stu)
            selected_tea_norm = self.norm(selected_tea)

            select_loss += F.l1_loss(selected_stu_norm, selected_tea_norm, reduction='mean') / b

            loss += (align_loss + select_loss)
            # ----------------------------------------------------------------------------
            selected_stu = selected_stu.reshape(b, c, h, w)
            selected_tea = selected_tea.reshape(b, c, h, w)

            cat_stu = torch.cat([cur_stu, aligned_stu, selected_stu], dim=1)
            cat_tea = torch.cat([cur_tea, aligned_tea, selected_tea], dim=1)

            conv_stu = self.__getattr__(f'conv_stu_{i}')
            conv_tea = self.__getattr__(f'conv_tea_{i}')

            aligned_selected_stu = (conv_stu(cat_stu) + cur_stu)
            aligned_selected_tea = (conv_tea(cat_tea) + cur_tea)

            # print(f'aligned selected stu shape: {aligned_selected_stu.shape}')
            out_stu.append(aligned_selected_stu)
            out_tea.append(aligned_selected_tea)
        
        return loss, out_stu, out_tea

    def norm(self, logit):
        mean = logit.mean(dim=-1, keepdims=True)
        std = logit.std(dim=-1, keepdims=True)
        return (logit - mean) / (1e-7 + std)

if __name__ == '__main__':
    x1 = torch.randn(1, 64, 96, 320).cuda()
    x2 = torch.randn(1, 128, 48, 160).cuda()
    x3 = torch.randn(1, 256, 24, 80).cuda()
    x4 = torch.randn(1, 512, 12, 40).cuda()

    stu = [x1, x2, x3, x4]
    tea = [x1, x2, x3, x4]

    m = AlignSelectLoss([64, 128, 256, 512]).cuda()

    loss, stu_feats, tea_feats = m(stu, tea)

