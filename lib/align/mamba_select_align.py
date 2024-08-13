import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.mamba.SSblock import MambaEncoder, Stem, ConvMamba

class SelfSelectProcess(nn.Module):
    def __init__(self, feat_size, in_dim, out_dim, inner_dim):
        super(SelfSelectProcess, self).__init__()
        self.me = MambaEncoder(feat_size=feat_size, in_channs=in_dim, inner_dim=inner_dim, out_dims=out_dim)

    def forward(self, x):
        '''
        Args:
            x: [B, C, H, W]

        Returns:
            [B, patch_nums, out_dim]
        '''
        return self.me(x)


class SelfSelectLoss2(nn.Module):
    def __init__(self, dim_list):
        super(SelfSelectLoss2, self).__init__()

        feat_size_list = [[96, 320], [48, 160], [24, 80]]
        out_dims = [16, 16*2, 16*4]
        inner_dims = [4, 4*2, 4*4]
        self.dim_list = dim_list

        if isinstance(dim_list, list):
            for i in range(len(dim_list)):
                select_process_stu= SelfSelectProcess(
                    feat_size=feat_size_list[i],
                    in_dim=dim_list[i],
                    out_dim=dim_list[i],
                    inner_dim=inner_dims[i]
                )

                select_process_tea = SelfSelectProcess(
                    feat_size=feat_size_list[i],
                    in_dim=dim_list[i],
                    out_dim=dim_list[i],
                    inner_dim=inner_dims[i]
                )
                self.__setattr__(f'select_process_stu_{i}', select_process_stu)
                self.__setattr__(f'select_process_tea_{i}', select_process_tea)

    def forward(self, rgb_feats, depth_feats):
        loss = 0.0
        if isinstance(rgb_feats, list):
            for i in range(len(self.dim_list)):
                cur_stu_feat = rgb_feats[i]
                cur_tea_feat = depth_feats[i]
                m_stu = self.__getattr__(f'select_process_stu_{i}')
                m_tea = self.__getattr__(f'select_process_tea_{i}')
                # [B, M, C]
                B = cur_stu_feat.shape[0]

                selected_stu_feat = m_stu(cur_stu_feat)
                selected_tea_feat = m_tea(cur_tea_feat)

                selected_stu_feat = selected_stu_feat.transpose(1, 2)
                selected_tea_feat = selected_tea_feat.transpose(1, 2)

                stu_feat_norm = self.norm(selected_stu_feat)
                tea_feat_norm = self.norm(selected_tea_feat)

                log_pred_stu = F.log_softmax(stu_feat_norm, dim=1)
                pred_tea = F.softmax(tea_feat_norm, dim=1)
                kl_loss = F.kl_div(log_pred_stu, pred_tea, reduction='none').sum(1).mean()

                l1_loss = F.l1_loss(stu_feat_norm, tea_feat_norm, reduction='mean') / B
                
                loss += (l1_loss + kl_loss)
        return loss

    def cosine_loss(self, stu_feat, tea_feat):
        return 1 - F.cosine_similarity(stu_feat, tea_feat, dim=-1)

    @staticmethod
    def norm(logit):
        mean = logit.mean(dim=-1, keepdims=True)
        std = logit.std(dim=-1, keepdims=True)
        return (logit - mean) / (1e-7 + std)


class SelfSelectLoss(nn.Module):
    def __init__(self, dim_list):
        super(SelfSelectLoss, self).__init__()
        feat_size_list = [[96, 320], [48, 160], [24, 80], [12, 40]]
        inner_dims = [4, 4 * 2, 4 * 4]
        # out_dims = [16, 16 * 2, 16 * 4]

        self.dim_list = dim_list

        for i in range(len(feat_size_list)):
            # patch_embed = Stem(feat_size_list[i], in_channs=dim_list[i], inner_dim=inner_dims[i], outer_dim=dim_list[i])
            # cur_feat = feat_size_list[i]
            conv_ssm = ConvMamba(dim_list[i])

            # self.__setattr__(f'patch_embed_{i}', patch_embed)
            self.__setattr__(f'conv_ssm_{i}', conv_ssm)


    def forward(self, rgb_feats, depth_feats):
        loss = 0.0

        for i in range(len(self.dim_list)):
            cur_stu_feat = rgb_feats[i]
            cur_tea_feat = depth_feats[i]

            B, C, H, W = cur_stu_feat.shape

            # [B, C, H, W] -> [B, C, H*W] -> [B, N, C]
            out_tokens_stu = cur_stu_feat.flatten(start_dim=2).permute(0, 2, 1)
            out_tokens_tea = cur_tea_feat.flatten(start_dim=2).permute(0, 2, 1)
            # patch_embed = self.__getattr__(f'patch_embed_{i}')
            # _, out_tokens_stu, _, _ = patch_embed(cur_stu_feat)
            # _, out_tokens_tea, _, _ = patch_embed(cur_tea_feat)

            conv_ssm = self.__getattr__(f'conv_ssm_{i}')
            feat_stu = self.norm(conv_ssm(out_tokens_stu))  # [B, N, C]
            feat_tea = self.norm(conv_ssm(out_tokens_tea))
            # print(f'feat stu shape: {feat_stu.shape}')

            feat_stu_reshape = torch.reshape(feat_stu.permute(0, 2, 1), (B, C, H, W))
            feat_tea_reshape = torch.reshape(feat_tea.permute(0, 2, 1), (B, C, H, W))

            cur_stu_feat = cur_stu_feat + feat_stu_reshape
            cur_tea_feat = cur_tea_feat + feat_tea_reshape

            rgb_feats[i] = cur_stu_feat
            depth_feats[i] = cur_tea_feat

            # print(f'rgb_{i} shape: {rgb_feats[i].shape}')

            l1_loss = F.l1_loss(feat_stu, feat_tea, reduction='mean') / B
            # print(f'cos loss: {cos_loss.shape}')
            loss += l1_loss
        return loss, rgb_feats, depth_feats


    def norm(self, logit):
        mean = logit.mean(dim=1, keepdims=True)
        std = logit.std(dim=1, keepdims=True)
        return (logit - mean) / (1e-7 + std)


if __name__ == '__main__':
    import numpy as np
    x1 = torch.randn(1, 64, 96, 320).cuda()
    x2 = torch.randn(1, 128, 48, 160).cuda()
    x3 = torch.randn(1, 256, 24, 80).cuda()
    x4 = torch.randn(1, 512, 12, 40).cuda()

    stu = [x1, x2, x3, x4]
    tea = [x1, x2, x3, x4]


    m = SelfSelectLoss2([64, 128, 256, 512]).cuda()
    out = m(stu, tea)
    print(f'out: {out[0]}')






