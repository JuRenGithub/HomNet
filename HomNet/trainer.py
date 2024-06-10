import os
import torch
import numpy as np
import sys
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import _LRScheduler
from HomNet.utils import print_result
from torch.utils.data import DataLoader


class WarmUpLR(_LRScheduler):
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]

def print_data(s):
    sys.stdout.write(str(s) + '\r')
    sys.stdout.flush()


class Trainer(object):
    def __init__(self, train_dataset, valid_dataset, test_dataset, model, config, save_path):
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.save_path = save_path
        self.model = model
        self.optim = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.l2)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optim, T_0=10)
        self.warmup_iter = config.warmup_iter
        self.warmup_scheduler = WarmUpLR(self.optim, self.warmup_iter)
        self.config = config
        self.device = config.device

        self.crit = torch.nn.CrossEntropyLoss()

    def batch_cuda(self, batch):
        for i in range(len(batch)):
            if isinstance(batch[i], torch.Tensor):
                batch[i] = batch[i].to(self.device)

    def evaluate(self, dataset):
        self.model.eval()
        loss_list = []
        data_loader = DataLoader(dataset, self.config.batch_size, shuffle=False)
        with torch.no_grad():
            TP = FP = TN = FN = 0
            scores_tensors = []
            labels_tensors = []
            for i, batch in enumerate(data_loader):
                self.batch_cuda(batch)
                x, x_cms, x_band, label = batch

                out = self.model(x, x_cms, x_band)
                loss = self.crit(out[0], label)
                    
                loss_list.append(loss.item())
                
                scores = torch.softmax(out[0], dim=-1)
                _, predicted = torch.max(out[0], dim=1)
                
                scores_tensors.append(scores[:, 1])
                labels_tensors.append(label)
                TP += ((predicted == 1) & (label == 1)).sum().item()
                FP += ((predicted == 1) & (label == 0)).sum().item()
                TN += ((predicted == 0) & (label == 0)).sum().item()
                FN += ((predicted == 0) & (label == 1)).sum().item()
            
            eval_loss = np.mean(loss_list)
            print('eval loss:', np.mean(eval_loss))
            
            merged_scores = torch.squeeze(torch.cat(scores_tensors)).cpu().numpy()
            merged_labels = torch.cat(labels_tensors).cpu().numpy()

            auc_roc = roc_auc_score(merged_labels, merged_scores)

        print('Roc auc: {:.6f}'.format(auc_roc))

        f1 = print_result(TP, FP, FN, TN)
        print('f1: {:.6f}'.format(f1))
        
        return eval_loss, auc_roc, f1

    def train(self):
        train_history = []   
        valid_history = []     
        best_auc = 0
        train_loader = DataLoader(self.train_dataset, self.config.batch_size, shuffle=True, num_workers=16)
        data_iter = iter(train_loader)
        iter_count = 0
        training_loss = []
        while iter_count < self.config.max_iter:
            try:
                batch = next(data_iter)
            except StopIteration as e:
                train_loader = DataLoader(self.train_dataset, self.config.batch_size, shuffle=True, num_workers=16)
                data_iter = iter(train_loader)
                batch = next(data_iter)
        
            self.batch_cuda(batch)
            x, x_cms, x_band, label = batch
            out = self.model(x, x_cms, x_band)
            loss = self.crit(out[0], label)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            training_loss.append(loss.item())

            if iter_count < self.warmup_iter:
                self.warmup_scheduler.step()
            
            iter_count += 1
            if iter_count % self.config.show_iter == 0:
                if iter_count > 1000:
                    self.scheduler.step()
                print()
                print("iter {:d} loss is: {:f}".format(iter_count, np.mean(training_loss)))
                train_history.append(np.mean(training_loss))
                training_loss = []
                # Compute eval metrics
                valid_loss, auc_roc, f1 = self.evaluate(self.valid_dataset)
                valid_history.append(valid_loss)
                if auc_roc > best_auc:
                    best_auc = auc_roc
                    count = 0
                    if self.config.log == 1:
                        if self.config.ft == 1:
                            model_save_path = os.path.join(self.save_path, f'fine_tuning')
                            if not os.path.exists(model_save_path):
                                os.makedirs(model_save_path)
                            self.model.save_ft_model(model_save_path)
                        else:
                            model_save_path = os.path.join(self.save_path, 'pretrain_model')
                            if not os.path.exists(model_save_path):
                                os.makedirs(model_save_path)
                            self.model.save_model(path=model_save_path)
                else:
                    count += 1
                    if count > self.config.early_stop:
                        print('early stop')
                        break
                if not self.config.eval_train == 1:
                    continue
        return train_history, valid_history, self.evaluate(self.test_dataset)

