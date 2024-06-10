import torch
import torch.nn as nn
from HomNet.common import CMSModel
import os


class HomNet(nn.Module):
    def __init__(self, config):
        super(HomNet, self).__init__()
        self.device = config.device
        self.name = config.name
        self.vec_dim = config.vec_dim
        self.seq_size = config.seq_size
        self.vec_n = 512 // config.seq_size

        self.model = CMSModel(config.emb_dim, self.vec_dim, self.vec_n, self.seq_size,
                              config.cell_num, config.n_head, config.n_layer, config.drop_out, config.device).to(config.device)
        layer_count = 0
        for p in self.parameters():
            if layer_count > config.frozen:
                break
            layer_count += 1
            p.requires_grad = False

    def forward(self, xs, x_cms=None, x_band=None, save_attn=False):
        out0, p_emb, attn_list0, attn_list1 = self.model(xs, x_cms, x_band, save_attn)

        return out0, p_emb, attn_list0, attn_list1

    def load_model(self, path='./model_dir/'):
        checkpoint = torch.load(os.path.join(path, f'{self.name}.pt'), map_location=lambda storage, loc: storage)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model = self.model.to(self.device)

    def load_ft_model(self, path='./model_dir/', ft_name=''):
        checkpoint = torch.load(os.path.join(path, f'{self.name}{ft_name}.pt'),
                                map_location=lambda storage, loc: storage)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model = self.model.to(self.device)

    def save_model(self, path='./model_dir/'):
        torch.save({'state_dict': self.model.state_dict()},
                   os.path.join(path, f'{self.name}.pt'))

    def save_ft_model(self, path='./model_dir/', ft_name=''):
        torch.save({'state_dict': self.model.state_dict()},
                   os.path.join(path, f'{self.name}{ft_name}.pt'))
