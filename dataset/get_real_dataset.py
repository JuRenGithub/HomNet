import numpy as np
from torch.utils.data import Dataset
import sys
sys.path.append('/home/user_name/HomNet') 
from dataset.data_utils import CMSDataset


def get_real_dataset(hos_name, rate):
    # return 3 CMSDataset: training set, valid set, test set
    return None, None, None


def get_demo_real_dataset(sample_num=2048):
    datas = np.random.rand(sample_num, 5, 4, 512).astype(np.float32)
    types = [
        np.random.randint(0, 2, (sample_num, )).astype(int),
        np.random.randint(0, 2, (sample_num, 24)).astype(np.float32),  # should be one-hot code in real dataset
        np.random.randint(0, 2, (sample_num, 5, 4)).astype(np.float32),  # also should be one-hot code
    ]

    return CMSDataset(datas[: -512], [types[i][:-512] for i in range(3)]), CMSDataset(datas[-512:], [types[i][-512:] for i in range(3)])