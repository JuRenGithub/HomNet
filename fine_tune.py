import torch
from model import Model
from train import train
from get_dataloader import get_loader
import os
import argparse

parser = argparse.ArgumentParser(description='for test')

parser.add_argument('--date', type=str, default='7-7')
parser.add_argument('--hos_name', type=str, default='hos1')
parser.add_argument('--save_path', type=str, default='save_path')
parser.add_argument('--name', type=str, default='pretrain_model')
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--emb_dim', type=int, default=128)
parser.add_argument('--head', type=int, default=4)
parser.add_argument('--layer', type=int, default=2)
parser.add_argument('--dp', type=float, default=0.)
parser.add_argument('--seq_size', type=int, default=32)
parser.add_argument('--vec_dim', type=int, default=64)
parser.add_argument('--eval', type=str, default='roc')
parser.add_argument('--lr', type=float, default=0.00001)
parser.add_argument('--l1', type=float, default=0.)
parser.add_argument('--l2', type=float, default=0.)
parser.add_argument('--epochs', type=int, default=100)


if __name__ == '__main__':
    args = parser.parse_args()
    ft_info = [args.date, f'ft_{args.hos_name}']

    batch_size = args.batch_size
    cell_num = 5
    emb_dim = args.emb_dim
    n_head = args.head
    n_layer = args.layer
    drop_out = args.dp
    seq_size = args.seq_size
    vec_dim = args.vec_dim


    device = torch.device(f'cuda:{args.cuda}')
    model = Model(device, args.name, cell_num, emb_dim, n_head, n_layer, drop_out, seq_size, vec_dim,
                  frozen=0)
    model.load_model(os.path.join(args.save_path, 'pretrain_model'))
    print('model ok')

    train_loader, test_loader = get_loader()

    weight = torch.Tensor([1, 10])
    crit = [torch.nn.CrossEntropyLoss(weight=weight).to(model.device),
            torch.nn.CrossEntropyLoss()]
    evaluate = args.eval
    optim = {'optim': 'adam', 'lr': args.lr, 'l1': args.l1, 'l2': args.l2, 're_norm': args.re_norm}
    lr_schedule = {'milestones': [5, 10, 15], 'gamma': 0.2}
    early_stop = 20

    train_loss, test_loss = train(train_loader, test_loader, model, crit, epochs=args.epochs, ft_info=ft_info,
                                  save_path=args.save_path, evaluate=evaluate, optim=optim, lr_schedule=lr_schedule,
                                  early_stop=early_stop)
