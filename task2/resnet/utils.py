import os
import math
import random
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import torch
from torch.optim.lr_scheduler import LambdaLR

def seed_all(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def show_img(input, size=(128, 128)):
    if isinstance(input, torch.Tensor):
        input = input.numpy()
    plt.imshow(Image.fromarray(input).resize(size))

class EarlyStopping():
    def __init__(self, patience=20, maximize=True, verbose=True, delta=0, model_name='model_param.pth'):
        self.patience = patience
        self.maximize = maximize
        self.verbose = verbose
        self.delta = delta
        self.model_name = model_name

        self._counter = 0
        self.best_score = None
        self.early_stop = False
        if self.maximize:
            self.val_max = -np.Inf
        else:
            self.val_min = np.Inf
    
    def __call__(self, val, model, model_path=None):
        score = val
        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(val, model, model_path)
        else:
            if self.maximize:
                if score < self.best_score  - self.delta:
                    self._counter += 1
                else:
                    self.best_score = score
                    self._save_checkpoint(val, model, model_path)
                    self._counter = 0
            else:
                if score > self.best_score + self.delta:
                    self._counter += 1      
                else:
                    self.best_score = score
                    self._save_checkpoint(val, model, model_path)
                    self._counter = 0    
            if self._counter >= self.patience:
                self.early_stop = True
    
    def _save_checkpoint(self, val, model, model_path=None):
        if model_path is None:
            file_path = os.path.join(os.getcwd(), self.model_name)
        else:
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            file_path = os.path.join(model_path, self.model_name)
        torch.save(model.state_dict(), file_path)
        if self.verbose:
            if self.maximize:
                print("Value Increase ({:.6f} --> {:.6f})".format(self.val_max, val))
            else:
                print("Value Decrease ({:.6f} --> {:.6f})".format(self.val_min, val))
        if self.maximize:
            self.val_max = val
        else:
            self.val_min = val

class Metrics():
    def __init__(self, metrics=None):
        self._metrics = metrics
        self._check_config()

    def _check_config(self):
        default_metrics = ["precision", "recall", "f1_score", "micro_f1", "macro_f1", "accuracy"]
        if isinstance(self._metrics, str):
            self._metrics = [self._metrics]
        for metric in self._metrics:
            if metric.lower() not in default_metrics:
                raise ValueError()
    
    def __call__(self, predict, label):
        predict = predict.cpu()
        label = label.cpu()
        res_dict = dict()
        for metric in self._metrics:
            metric = metric.lower()
            if metric == 'precision':
                res_dict[metric] = precision_score(y_true=label, y_pred=predict, average='macro')
            elif metric == 'recall':
                res_dict[metric] = recall_score(y_true=label, y_pred=predict, average='macro')
            elif metric == 'accuracy':
                res_dict[metric] = accuracy_score(y_true=label, y_pred=predict)
            elif metric == 'f1_score':
                res_dict['micro_f1'] = f1_score(y_true=label, y_pred=predict, average='micro')
                res_dict['macro_f1'] = f1_score(y_true=label, y_pred=predict, average='macro')
            elif metric == 'micro_f1':
                res_dict['micro_f1'] = f1_score(y_true=label, y_pred=predict, average='micro')
            elif metric == 'macro_f1':
                res_dict['macro_f1'] = f1_score(y_true=label, y_pred=predict, average='macro')
        return res_dict
    
class Warmup(LambdaLR):
    def __init__(self, optimizer, warmup_steps, t_total, cycles=0.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(Warmup, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)
    
    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0, 0.5 * (1 + math.cos(math.pi * float(self.cycles) * 2 * progress)))