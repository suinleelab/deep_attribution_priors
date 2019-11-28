import torch
import numpy as np
from IPython.core.debugger import set_trace


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, minimize=True):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each increase in stopping count
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_metric_min = np.Inf
        self.delta = delta
        self.minimize = minimize

    def __call__(self, val_metric, model):

        score = val_metric
        if self.minimize:
            score *= -1

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_metric, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_metric, model)
            self.counter = 0

    def save_checkpoint(self, val_metric, model):
        '''Saves model when validation metric improves.'''
        print(f'Validation loss decreased ({self.val_metric_min:.6f} --> {val_metric:.6f}).  Saving model ...')
        if type(model) == list:
            for i, model in enumerate(model):
                torch.save(model.state_dict(), 'checkpoint_{}.pt'.format(i))
        else:
            torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_metric_min = val_metric