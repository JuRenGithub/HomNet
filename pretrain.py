import joblib
from HomNet.HomNet import HomNet
from HomNet.trainer import Trainer
from dataset.get_pretrain_dataset import get_pretrain_dataset
# from old_code.tools import MyLog
import os
import argparse


def main(config, save_path):
    model = HomNet(config)
    print('load model ok')

    data_root = './dataset/pretrain_data'
    train_dataset, test_dataset = get_pretrain_dataset(data_root, config.with_band)
    valid_dataset = test_dataset
    print('load dataset ok')

    trainer = Trainer(train_dataset, valid_dataset, test_dataset, model, config, save_path)
    train_loss, test_loss = trainer.train()

    if config.log == 1:
        joblib.dump([train_loss, test_loss], os.path.join(save_path, 'loss.pkl'))


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--cell_num', type=int, default=5)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--drop_out', type=float, default=0.5)
parser.add_argument('--early_stop', type=int, default=5)
parser.add_argument('--emb_dim', type=int, default=128)
parser.add_argument('--eval_train', type=int, default=0)
parser.add_argument('--frozen', type=int, default=0)
parser.add_argument('--ft', type=int, default=0)
parser.add_argument('--log', type=int, default=0)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--l2', type=float, default=1e-3)
parser.add_argument('--max_iter', type=int, default=50000)
parser.add_argument('--n_head', type=int, default=4)
parser.add_argument('--n_layer', type=int, default=2)
parser.add_argument('--name', type=str, default='HomNet')
parser.add_argument('--seq_size', type=int, default=8)
parser.add_argument('--show_iter', type=int, default=500)
parser.add_argument('--warmup_iter', type=int, default=500)
parser.add_argument('--with_band', type=int, default=1)
parser.add_argument('--vec_dim', type=int, default=64)


if __name__ == '__main__':
    config = parser.parse_args()
    save_path = './save_dir'  # change to your own save_path
    main(config, save_path)    
