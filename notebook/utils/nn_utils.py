import numpy as np
import torch


class EarlyStopping:
    """
    ref: https://github.com/Bjarten/early-stopping-pytorch
    """
    def __init__(self, patience=2, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model, save_name):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, save_name)
        elif score < self.best_score:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} '
            #      'out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, save_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, save_name):
        if self.verbose:
            print(f'Validation loss decreased ('
                  '{self.val_loss_min:.5f} --> {val_loss:.5f}'
                  ').  Saving model ...')
            print("Save model: {}".format(save_name))
        torch.save(model.state_dict(), save_name)
        self.val_loss_min = val_loss
