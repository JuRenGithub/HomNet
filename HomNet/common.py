import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from einops.layers.torch import Rearrange


class BasicFF(nn.Module):
    def __init__(self, in_dim, hidden_dim, drop_out):
        super(BasicFF, self).__init__()
        self.net1 = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim, bias=True),
            nn.GELU(),
            nn.Dropout(drop_out)
        )
        self.net2 = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, in_dim, bias=True),
            nn.Dropout(drop_out)
        )

    def forward(self, x):
        out = self.net1(x)
        out = self.net2(out)
        out = F.gelu(out + x)

        return out


class CMSEncoder(nn.Module):
    def __init__(self, vec_dim, vec_n, drop_out):
        super(CMSEncoder, self).__init__()
        self.ff1 = nn.Sequential(
            nn.LayerNorm(vec_dim),
            BasicFF(vec_dim, vec_dim * 2, drop_out)
        )
        self.ff2 = nn.Sequential(
            Rearrange('b n d -> b d n'),
            nn.LayerNorm(vec_n),
            BasicFF(vec_n, vec_n * 2, drop_out),
            Rearrange('b d n -> b n d')
        )

    def forward(self, x):
        out = self.ff1(x)
        out = self.ff2(out)
        out = F.gelu(out + x)

        return out


class SeqMerge(nn.Module):
    def __init__(self, vec_dim, sub_seq_size):
        super(SeqMerge, self).__init__()
        self.conv_mg = nn.Sequential(
            nn.Conv1d(2, vec_dim, kernel_size=(sub_seq_size,), stride=(sub_seq_size,), bias=False),
            nn.BatchNorm1d(vec_dim),
            nn.ReLU(),
            Rearrange('b c l -> b l c')
        )

    def forward(self, x):
        out = self.conv_mg(x)
        return out


class Attention(nn.Module):
    def __init__(self, d, drop_out):
        super(Attention, self).__init__()
        self.d = d

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask=None):
        _attn = torch.matmul(q / self.d, k.transpose(2, 3))
        if mask is not None:
            attn = _attn.masked_fill(mask, -1e9)
        else:
            attn = _attn
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)

        return out, attn


class MLP(nn.Module):
    def __init__(self, vec_dim, hidden_dim, drop_out):
        super(MLP, self).__init__()
        self.lin = nn.Sequential(
            nn.Linear(vec_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_out),
            nn.Linear(hidden_dim, vec_dim),
            nn.GELU(),
            nn.Dropout(drop_out)
        )

    def forward(self, x):
        _out = self.lin(x)
        out = _out + x
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, vec_dim, k_v_dim, drop_out):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.k_v_dim = k_v_dim

        self.w_qs = nn.Linear(vec_dim, n_head * k_v_dim, bias=False)
        self.w_ks = nn.Linear(vec_dim, n_head * k_v_dim, bias=False)
        self.w_vs = nn.Linear(vec_dim, n_head * k_v_dim, bias=False)

        self.fc1 = nn.Sequential(
            nn.Linear(n_head * k_v_dim, vec_dim, bias=False),
            nn.GELU(),
            nn.Dropout(drop_out)
        )
        self.attention = Attention(k_v_dim ** 0.5, drop_out)
        self.mlp = MLP(vec_dim * 2, vec_dim, drop_out)
        self.fc2 = nn.Sequential(
            nn.Linear(vec_dim * 2, vec_dim, bias=False),
            nn.GELU(),
            nn.Dropout(drop_out)
        )

    def forward(self, q_vec: torch.Tensor, k_vec: torch.Tensor, mask=None):
        res_q = q_vec
        batch_size, v_n, k_n = q_vec.size(0), q_vec.size(1), k_vec.size(1)

        q = self.w_qs(q_vec).view(batch_size, v_n, self.n_head, self.k_v_dim)
        k = self.w_ks(k_vec).view(batch_size, k_n, self.n_head, self.k_v_dim)
        v = self.w_vs(k_vec).view(batch_size, k_n, self.n_head, self.k_v_dim)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        q_out, attn = self.attention(q, k, v, mask)
        q_out = q_out.transpose(1, 2).contiguous().view(batch_size, v_n, -1)
        q_out = self.fc1(q_out)
        _out = torch.cat((res_q, q_out), dim=2)

        out = self.mlp(_out)
        out = self.fc2(out)

        return out, attn


class HomMerge(nn.Module):
    def __init__(self, vec_n, vec_dim, emb_dim, drop_out):
        super(HomMerge, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Sequential(
            nn.Conv1d(in_channels=vec_n, out_channels=vec_n // 2, kernel_size=(1,), bias=False),
            nn.ReLU(),
            nn.Conv1d(in_channels=vec_n // 2, out_channels=vec_n, kernel_size=(1,), bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        self.lin = nn.Sequential(
            nn.Linear(vec_dim, emb_dim),
            nn.GELU(),
            nn.Dropout(drop_out)
        )

    def forward(self, vec):
        avg_out = self.fc(self.avg_pool(vec))
        max_out = self.fc(self.max_pool(vec))
        out = avg_out + max_out
        att = self.sigmoid(out).view(vec.size(0), 1, -1)
        out = torch.squeeze(torch.matmul(att, vec), 1)
        emb = self.lin(out)
        return emb


class HomoBlock(nn.Module):
    def __init__(self, vec_dim, vec_n, emb_dim, n_head, n_layer, device, drop_out):
        super(HomoBlock, self).__init__()
        self.device = device
        self.n_head = n_head
        self.vec_dim = vec_dim
        self.k_v_dim = 16
        self.vec_n = vec_n
        self.pos_code = self.get_pos_code()
        self.pos_encoder = nn.Sequential(
            nn.Linear(self.vec_n, self.vec_dim),
            nn.Dropout(drop_out),
        )
        self.attention_layers = nn.ModuleList([
            MultiHeadAttention(self.n_head, self.vec_dim, self.k_v_dim, drop_out)
            for _ in range(n_layer)
        ])
        self.to_emb = HomMerge(vec_n, vec_dim * 2, emb_dim, drop_out)

    def get_pos_code(self):
        pos_code = torch.zeros((self.vec_n, self.vec_n), dtype=torch.float32, device=self.device)
        for i in range(self.vec_n):
            pos_code[i][i] = 1
        pos_code = pos_code.unsqueeze(0)
        return pos_code

    def forward(self, cms_emb0: torch.Tensor, cms_emb1: torch.Tensor, save_attn=False):
        att_list0 = []
        att_list1 = []
        pos = self.pos_encoder(self.pos_code)
        vec0 = cms_emb0 + pos
        vec1 = cms_emb1 + pos

        for att_layer in self.attention_layers:
            vec0, att0 = att_layer(vec0, vec1)
            vec1, att1 = att_layer(vec1, vec0)
            if save_attn:
                att_list0.append(att0)
                att_list1.append(att1)
        vec = torch.cat((vec0, vec1), dim=-1)

        emb = self.to_emb(vec)
        return emb, att_list0, att_list1


class CMSBlock(nn.Module):
    def __init__(self, vec_dim, vec_n, sub_seq_size, drop_out):
        super(CMSBlock, self).__init__()
        self.vec_n = vec_n
        self.to_vec = SeqMerge(vec_dim, sub_seq_size)
        self.info_net = nn.Linear(28, vec_dim * vec_n, bias=True)
        self.cms_encoder = CMSEncoder(vec_dim, vec_n, drop_out)

    def forward(self, x: torch.Tensor, x_cms=None, x_band=None):

        b_n = x.size(0)
        mg = self.to_vec(x)
        if x_cms is not None:
            x_info = self.info_net(torch.cat((x_cms, x_band), dim=1)).view(b_n, self.vec_n, -1)
            mg = mg + x_info

        out = self.cms_encoder(mg)

        return out


class BagBlock(nn.Module):
    def __init__(self, cell_num, emb_dim, drop_out=0):
        super(BagBlock, self).__init__()
        self.cell_num = cell_num

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.se_fc = nn.Sequential(
            nn.Linear(cell_num, cell_num // 2),
            nn.GELU(),
            nn.Linear(cell_num // 2, cell_num),
            nn.Sigmoid()
        )

        self.mlp = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, emb_dim // 2),
            nn.GELU(),
            nn.Dropout(drop_out),
            nn.Linear(emb_dim // 2, emb_dim),
            nn.GELU(),
            nn.Dropout(drop_out)
        )

    def forward(self, hom_embs):
        fuse_cells_data = torch.stack(hom_embs, dim=1)
        avg_att = self.avg_pool(fuse_cells_data).view(fuse_cells_data.size(0), -1)
        max_att = self.max_pool(fuse_cells_data).view(fuse_cells_data.size(0), -1)
        avg_att = self.se_fc(avg_att).view(fuse_cells_data.size(0), 1, -1)
        max_att = self.se_fc(max_att).view(fuse_cells_data.size(0), 1, -1)
        att_cells_data = avg_att + max_att
        fuse_data = torch.matmul(att_cells_data, fuse_cells_data)
        fuse_data = torch.squeeze(fuse_data, dim=1)
        return self.mlp(fuse_data)


class CMSModel(nn.Module):
    def __init__(self, emb_dim, vec_dim, vec_n, seq_size, cell_num, n_head, n_layer, drop_out, device):
        super(CMSModel, self).__init__()
        self.cell_num = cell_num
        self.cms_block = CMSBlock(vec_dim, vec_n, seq_size, drop_out)
        self.homo_block = HomoBlock(vec_dim, vec_n, emb_dim, n_head, n_layer, device, drop_out)
        self.bag_block = BagBlock(cell_num, emb_dim, drop_out)
        self.cls_head = nn.Sequential(
            nn.Linear(emb_dim, 2)
        )

    def forward(self, x, x_cms, x_band, save_attn=False):
        home_embs = []
        attn_list0, attn_list1 = [], []
        for i in range(self.cell_num):
            cms_emb0 = self.cms_block(x[:, i, :2], x_cms, None if x_band is None else x_band[:, i])
            cms_emb1 = self.cms_block(x[:, i, 2:], x_cms, None if x_band is None else x_band[:, i])
            home_emb, attn_l0, attn_l1 = self.homo_block(cms_emb0, cms_emb1, save_attn)
            home_embs.append(home_emb)
            if save_attn:
                attn_list0.append(attn_l0)
                attn_list1.append(attn_l1)

        emb = self.bag_block(home_embs)
        return self.cls_head(emb), emb, attn_list0, attn_list1

