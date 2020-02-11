import torch
import numpy as np
import time
import datetime
import seaborn as sns
import matplotlib.pyplot as plt


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, minimize=True):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints out logging messages for early stopping/model saving
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
        elif score <= self.best_score + self.delta:
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
        if self.verbose:
            print(f'Validation loss decreased ({self.val_metric_min:.6f} --> {val_metric:.6f}).  Saving model ...')
        if type(model) == list:
            for i, model in enumerate(model):
                torch.save(model.state_dict(), 'checkpoint_{}.pt'.format(i))
        else:
            torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_metric_min = val_metric


def train(model, optimizer, criterion, train_loader, valid_loader, test_loader, patience):
    print("Beginning model training at {}".format(datetime.datetime.now()))

    early_stopping = EarlyStopping(patience=patience, minimize=False)
    epoch = 0

    while True:
        start_time = time.time()

        model.train()
        for X_batch, y_batch in train_loader:
            y_batch = y_batch.squeeze()
            X_batch = torch.FloatTensor(X_batch).cuda()
            y_batch = torch.LongTensor(y_batch).cuda()

            optimizer.zero_grad()

            output = model(X_batch)
            loss_train = criterion(output, y_batch)
            loss_train.backward()
            optimizer.step()

        model.eval()
        for X_batch, y_batch in valid_loader:
            y_batch = y_batch.squeeze()
            X_batch = torch.FloatTensor(X_batch).cuda()
            y_batch = torch.LongTensor(y_batch).cuda()

            output = model(X_batch)
            loss_valid = criterion(output, y_batch).item()
            class_probabilities = torch.nn.Softmax()(output)
            predicted_classes = torch.argmax(class_probabilities, dim=1)
            correct_predictions = (predicted_classes == y_batch).float()
            accuracy = (correct_predictions.sum() / correct_predictions.shape[0]).item()
            end_time = time.time()
            epoch_time = end_time - start_time

        if epoch % 10 == 0:
            print("Epoch {} completed in {} secs with validation loss, accuracy {:.4f},{:.4f}".format(epoch, epoch_time,
                                                                                                      loss_valid,
                                                                                                      accuracy))
        epoch += 1
        # early_stopping needs the validation loss to check if it has decreased,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(accuracy, model)

        if early_stopping.early_stop:
            print("Early stopping on epoch {}".format(epoch))
            break

    model.load_state_dict(torch.load('checkpoint.pt'))
    for X_batch, y_batch in test_loader:
        y_batch = y_batch.squeeze()
        X_batch = torch.FloatTensor(X_batch).cuda()
        y_batch = torch.LongTensor(y_batch).cuda()

        output = model(X_batch)
        class_probabilities = torch.nn.Softmax()(output)
        predicted_classes = torch.argmax(class_probabilities, dim=1)
        correct_predictions = (predicted_classes == y_batch).float()
        accuracy = (correct_predictions.sum() / correct_predictions.shape[0]).item()
    return accuracy


def train_with_learned_prior(
        f1,
        f2,
        f1_optimizer,
        f2_optimizer,
        criterion,
        train_loader,
        valid_loader,
        test_loader,
        patience,
        explainer,
        prior_info):
    print("Beginning model training at {}".format(datetime.datetime.now()))

    early_stopping = EarlyStopping(patience=patience, minimize=False)
    epoch = 0

    while True:
        start_time = time.time()
        f1.train()
        f2.train()

        for X_batch, y_batch in train_loader:
            y_batch = y_batch.squeeze()
            X_batch = torch.FloatTensor(X_batch).cuda()
            y_batch = torch.LongTensor(y_batch).cuda()

            f1_optimizer.zero_grad()
            f2_optimizer.zero_grad()

            output = f1(X_batch)
            classification_loss = criterion(output, y_batch)

            eg = explainer.shap_values(f1, X_batch, sparse_labels=y_batch)
            prior_differences = f2(prior_info).squeeze()

            # This works pretty well! Why?
            prior_loss = (prior_differences - eg).abs().mean()
            train_loss = classification_loss + prior_loss

            train_loss.backward()
            f1_optimizer.step()
            f2_optimizer.step()

        f1.eval()
        for X_batch, y_batch in valid_loader:
            y_batch = y_batch.squeeze()
            X_batch = torch.FloatTensor(X_batch).cuda()
            y_batch = torch.LongTensor(y_batch).cuda()
            output = f1(X_batch)
            loss_valid = criterion(output, y_batch)

            class_probabilities = torch.nn.Softmax()(output)
            predicted_classes = torch.argmax(class_probabilities, dim=1)
            correct_predictions = (predicted_classes == y_batch).float()
            accuracy = (correct_predictions.sum() / correct_predictions.shape[0]).item()
            end_time = time.time()
            epoch_time = end_time - start_time

        if epoch % 10 == 0:
            print("Epoch {} completed in {} secs with test loss, accuracy {:.4f},{:.4f}".format(epoch, epoch_time,
                                                                                                loss_valid, accuracy))

        # early_stopping needs the validation loss to check if it has decreased,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(accuracy, [f1, f2])
        epoch += 1
        if early_stopping.early_stop:
            print("Early stopping on epoch {}".format(epoch))
            break

    f1.load_state_dict(torch.load('checkpoint_0.pt'))
    f2.load_state_dict(torch.load('checkpoint_1.pt'))

    for X_batch, y_batch in test_loader:
        y_batch = y_batch.squeeze()
        X_batch = torch.FloatTensor(X_batch).cuda()
        y_batch = torch.LongTensor(y_batch).cuda()

        output = f1(X_batch)
        class_probabilities = torch.nn.Softmax()(output)
        predicted_classes = torch.argmax(class_probabilities, dim=1)
        correct_predictions = (predicted_classes == y_batch).float()
        accuracy = (correct_predictions.sum() / correct_predictions.shape[0]).item()
    return accuracy

def metafeature_pdp(
    metafeatures,
    feature_to_alter,
    meta_range,
    color,
    model,
    xlabel="Predicted Attribution Value",
    ylabel="Metafeature",
):
    sns.set_style("darkgrid")
    meta_copy = metafeatures.copy()
    predicted_attributions_new_meta = []
    
    for new_meta_val in meta_range:
        meta_copy[feature_to_alter] = new_meta_val
        predicted_attributions_altered_feature = model(torch.FloatTensor(meta_copy.values).cuda()).abs()
        predicted_attributions_new_meta.append(predicted_attributions_altered_feature.mean().item())
    plt.plot(meta_range, predicted_attributions_new_meta, color=color)
    plt.xlabel(xlabel, fontsize=14)
    plt.xticks(fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.yticks(fontsize=14)