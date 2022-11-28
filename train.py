import os
import torch
import numpy as np
import sys
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter


class WarmUpLR(_LRScheduler):
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]

def print_data(s):
    sys.stdout.write(str(s) + '\r')
    sys.stdout.flush()


def train(train_loader, test_loader, model, crit, epochs=200, ft_info=None, save_path='./save',
          evaluate='p', optim=None, lr_schedule=None, early_stop=10, wp_iter=500):
    best_e = 0.0
    count = 0

    optimizer = torch.optim.AdamW(model.parameters(), lr=optim['lr'], weight_decay=optim['l2'])
    train_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_schedule['milestones'],
                                                           gamma=lr_schedule['gamma'])
    warm_iteration = wp_iter
    warmup_scheduler = WarmUpLR(optimizer, warm_iteration)
    for epoch in range(epochs):
        training_loss = []
        test_loss = []
        model.train()
        loader = train_loader
        for i, data in enumerate(loader):
            shuffle_ix = np.random.permutation(np.arange(5))
            data[0] = np.swapaxes(data[0], 0, 1)
            data[2] = np.swapaxes(data[2], 0, 1)
            data[0] = data[0][shuffle_ix]
            data[2] = data[2][shuffle_ix]
            data[0] = np.swapaxes(data[0], 0, 1)
            data[2] = np.swapaxes(data[2], 0, 1)

            x, x_cms, x_band, label, ab_label, y_info = data
            if x.device != model.device:
                x_band = x_band.to(model.device)
                x = x.to(model.device)
                x_cms = x_cms.to(model.device)
                label = label.to(model.device)
                ab_label = ab_label.to(model.device)

            optimizer.zero_grad()
            out = model(x, x_cms, x_band)
            loss = crit[0](out[0], label)
            loss = loss
            loss.backward()
            optimizer.step()
            training_loss.append(loss.item())
            if epoch == 0 and (i < warm_iteration):
                warmup_scheduler.step()

        train_scheduler.step()
        print()
        print("epoch {:d} loss is: {:f}".format(epoch, np.mean(training_loss)))

        model.eval()
        with torch.no_grad():
            TP = FP = TN = FN = 0
            scores_tensors = []
            labels_tensors = []
            for i, data in enumerate(test_loader):
                x, x_cms, x_band, label, ab_label, y_info = data
                if x.device != model.device:
                    x = x.to(model.device)
                    x_cms = x_cms.to(model.device)
                    label = label.to(model.device)
                    ab_label = ab_label.to(model.device)
                    x_band = x_band.to(model.device)

                out = model(x, x_cms, x_band)
                # loss
                loss = crit[0](out[0], label)
                if model.struct:
                    # classify structural abnormal type
                    loss = loss + 0.3 * crit[1](out[1], ab_label)

                test_loss.append(loss.item())

                scores = torch.softmax(out[0], dim=-1)
                _, predicted = torch.max(out[0], dim=1)

                scores_tensors.append(scores[:, 1])
                labels_tensors.append(label)
                TP += ((predicted == 1) & (label == 1)).sum().item()
                FP += ((predicted == 1) & (label == 0)).sum().item()
                TN += ((predicted == 0) & (label == 0)).sum().item()
                FN += ((predicted == 0) & (label == 1)).sum().item()
            print('test loss:', np.mean(test_loss))

            merged_scores = torch.squeeze(torch.cat(scores_tensors)).cpu().numpy()
            merged_labels = torch.cat(labels_tensors).cpu().numpy()
            precision, recall, thresholds = precision_recall_curve(merged_labels, merged_scores)
            pr_auc = auc(recall, precision)
            roc_auc = roc_auc_score(merged_labels, merged_scores)
            try:
                precision_at_80_recall = max(p for p, r in zip(precision, recall) if r >= 0.8)
            except:
                precision_at_80_recall = 0.0

                # f1: beta = 1; f2: beta =2
        print("Test positive precision @ 0.8 recall: {:.6f}".format(precision_at_80_recall))
        print('Test pr auc: {:.6f}'.format(pr_auc))
        print('Test roc auc: {:.6f}'.format(roc_auc))
        if evaluate == 'p':
            cur_e = precision_at_80_recall
        elif evaluate == 'pr':
            cur_e = pr_auc
        else:
            cur_e = roc_auc

        if cur_e > best_e:
            best_e = cur_e
            count = 0
            # save model
            if ft_info is not None:
                model_save_path = os.path.join(save_path, f'fine_tuning/{model.name}')
                if not os.path.exists(model_save_path):
                    os.makedirs(model_save_path)
                model.save_ft_model(model_save_path, ft_name=f'{ft_info[0]}_{ft_info[1]}')
            else:
                model_save_path = os.path.join(save_path, 'pretrain_model')
                if not os.path.exists(model_save_path):
                    os.makedirs(model_save_path)
                model.save_model(path=model_save_path)
        else:
            count += 1
            if count > early_stop:
                break