import torch
from HomNet.HomNet import HomNet
from HomNet.trainer import Trainer
from dataset.get_real_dataset import get_real_dataset
# from old_code.tools import MyLog
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--cell_num', type=int, default=5)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--drop_out', type=float, default=0.5)
parser.add_argument('--early_stop', type=int, default=5)
parser.add_argument('--emb_dim', type=int, default=128)
parser.add_argument('--eval_train', type=int, default=0)
parser.add_argument('--frozen', type=int, default=0)
parser.add_argument('--ft', type=int, default=1)
parser.add_argument('--hos', type=str, default='TestHos')
parser.add_argument('--log', type=int, default=0)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--l2', type=float, default=1e-3)
parser.add_argument('--max_iter', type=int, default=50000)
parser.add_argument('--n_head', type=int, default=4)
parser.add_argument('--n_layer', type=int, default=2)
parser.add_argument('--name', type=str, default='HomNet')
parser.add_argument('--seq_size', type=int, default=8)
parser.add_argument('--show_iter', type=int, default=250)
parser.add_argument('--hos', type=str)
parser.add_argument('--warmup_iter', type=int, default=500)
parser.add_argument('--with_band', type=int, default=1)
parser.add_argument('--vec_dim', type=int, default=64)
parser.add_argument('--run_time', type=int, default=1)


def main(config, save_path):
    model = HomNet(config)
    model_path = './pretrain_model'
    model.load_model(model_path)
    print('load model ok')

    train_dataset, valid_dataset, test_dataset = get_real_dataset(config.hos, 0.2)
    print('load dataset ok', config.hos)

    trainer = Trainer(train_dataset, valid_dataset, test_dataset, model, config, save_path)
    train_loss, valid_loss, test_result = trainer.train()


if __name__ == '__main__':
    config = parser.parse_args()
    save_path = './save_dir'
    for i in range(config.run_time-1):
        print('begin', i)
        main(config, save_path)   
        print('end', i)