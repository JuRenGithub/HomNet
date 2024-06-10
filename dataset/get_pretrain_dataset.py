import numpy as np
from torch.utils.data import Dataset
import sys
sys.path.append('/home/user_name/HomNet') 
from dataset.data_utils import CMSDataset


def get_pretrain_dataset(data_root, with_band=False):
    # implement with dataset you use
    # reture: train_dataset, test_dataset (or valid dataset, just for pretrain)
    return None, None  # CMSDataset


def get_demo_pretrain_dataset(sample_num=2048):
    datas = np.random.rand(sample_num, 5, 4, 512).astype(np.float32)
    types = [
        np.random.randint(0, 2, (sample_num, )).astype(int),
        np.random.randint(0, 2, (sample_num, 24)).astype(np.float32),  # should be one-hot code in real dataset
        np.random.randint(0, 2, (sample_num, 5, 4)).astype(np.float32),  # also should be one-hot code
    ]

    return CMSDataset(datas[: -512], [types[i][:-512] for i in range(3)]), CMSDataset(datas[-512:], [types[i][-512:] for i in range(3)])