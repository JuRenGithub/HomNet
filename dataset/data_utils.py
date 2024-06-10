import torch
from torch.utils.data import Dataset

class CMSDataset(Dataset):
    def __init__(self, datas, types, with_band=True):
        # datas: [n, 5, 4, 512]
        # types: [3, n*], [label, cms_type, band_type]
        # cms_type: one-hot code, shape: (n, 24)
        # band_type: one-hot code, shape: (n, 5, 4)
        self.with_band = with_band
        self.datas = torch.from_numpy(datas).type(torch.float32)
        self.labels, self.cms_types, self.band_types = types


    def __len__(self):        
        return len(self.labels)

    def __getitem__(self, idx):
        return self.datas[idx], self.cms_types[idx], self.band_types[idx], self.labels[idx]
    
    