import logging
import data
import datetime
import time
from utils import EarlyStopping
from models import MLP, LinearModel
from torch import optim
from sklearn.preprocessing import StandardScaler
from torch.nn import MSELoss
from torch.utils.data import DataLoader
import numpy as np
import torch
import os
import pickle
import egexplainer
import random
import pandas as pd
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--drug', help='Which drug to train on', type=str, required=True)
args = parser.parse_args()

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

# To keep things as reproducible as possible
seed = 1017
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Stops too many cores from being used by Torch
torch.set_num_threads(1)

# Load Data
response_data = data.ExVivoDrugData.load_data()
merge_data = data.MergeData.load_data()

# Reduce set of genes to those in both our rna-seq data and in the MERGE data
overlapping_genes = list(set(response_data.X.columns).intersection(merge_data.data.columns))
merge_data.data = merge_data.data[overlapping_genes]
response_data.X = response_data.X[["patient_id"] + response_data.drug_columns + overlapping_genes]

# Find drug with most number of samples
max_patients = 0
drug_with_max_patients = None
drug_col = "drug_{}".format(args.drug)

patience=100
criterion = MSELoss()

def train(model, optimizer):
    print("Beginning model training at {}".format(datetime.datetime.now()))
    early_stopping = EarlyStopping(patience=patience)
    epoch = 0
    while True:
        start_time = time.time()
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            X_batch = X_batch.float().cuda()
            y_batch = y_batch.float().cuda()

            output = model(X_batch)
            loss_train = criterion(output, y_batch)

            loss_train.backward()
            optimizer.step()

        model.eval()
        for X_batch, y_batch in valid_loader:
            X_batch = X_batch.float().cuda()
            y_batch = y_batch.float().cuda()

            output = model(X_batch)
            loss_valid = criterion(output, y_batch).item()
            end_time = time.time()
            epoch_time = end_time - start_time

        if epoch % 50 == 0:
            logging.info("Epoch {} completed in {} secs with valid loss {:.4f}".format(epoch, epoch_time, loss_valid))

        early_stopping(loss_valid, model)

        if early_stopping.early_stop:
            print("Early stopping on epoch {}".format(epoch))
            break
        epoch += 1
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.float().cuda()
        y_batch = y_batch.float().cuda()

        output = model(X_batch)
        loss_test = criterion(output, y_batch).item()
    return loss_test

def train_with_learned_prior(f1, f2, f1_optimizer, f2_optimizer, prior_info):
    print("Beginning model training at {}".format(datetime.datetime.now()))
    early_stopping = EarlyStopping(patience=patience)
    epoch = 0
    while True:
        start_time = time.time()
        f1.train()
        for X_batch, y_batch in train_loader:
            f1_optimizer.zero_grad()
            f2_optimizer.zero_grad()

            X_batch = X_batch.float().cuda()
            y_batch = y_batch.float().cuda()

            output = f1(X_batch)

            eg = APExp.shap_values(f1, X_batch).abs()
            prior_differences = f2(prior_info).squeeze()
            prior_loss = (prior_differences - eg).abs().mean()

            loss_train = criterion(output, y_batch) + prior_loss

            loss_train.backward()
            f1_optimizer.step()
            f2_optimizer.step()

        f1.eval()
        for X_batch, y_batch in valid_loader:
            X_batch = X_batch.float().cuda()
            y_batch = y_batch.float().cuda()

            output = f1(X_batch)
            loss_valid = criterion(output, y_batch).item()
            end_time = time.time()
            epoch_time = end_time - start_time

        if epoch % 20 == 0:
            logging.info("Epoch {} completed in {} secs with valid loss {:.4f}".format(epoch, epoch_time, loss_valid))

        early_stopping(loss_valid, [f1, f2])

        if early_stopping.early_stop:
            print("Early stopping on epoch {}".format(epoch))
            break
        epoch += 1
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.float().cuda()
        y_batch = y_batch.float().cuda()

        output = f1(X_batch)
        loss_test = criterion(output, y_batch).item()
    return loss_test

def train_with_merge_algo(f1, f2, f1_optimizer, f2_optimizer, prior_info):
    print("Beginning model training at {}".format(datetime.datetime.now()))
    early_stopping = EarlyStopping(patience=patience)
    epoch = 0
    while True:
        start_time = time.time()
        f1.train()
        for X_batch, y_batch in train_loader:
            f1_optimizer.zero_grad()
            f2_optimizer.zero_grad()

            X_batch = X_batch.float().cuda()
            y_batch = y_batch.float().cuda()

            output = f1(X_batch)

            prior_differences = f2(prior_info).squeeze()
            prior_loss = (prior_differences - f1.layers[0].weight).abs().mean()

            loss_train = criterion(output, y_batch) + prior_loss

            loss_train.backward()
            f1_optimizer.step()
            f2_optimizer.step()

        f1.eval()
        for X_batch, y_batch in valid_loader:
            X_batch = X_batch.float().cuda()
            y_batch = y_batch.float().cuda()

            output = f1(X_batch)
            loss_valid = criterion(output, y_batch).item()
            end_time = time.time()
            epoch_time = end_time - start_time

        if epoch % 10 == 0:
            logging.info("Epoch {} completed in {} secs with valid loss {:.4f}".format(epoch, epoch_time, loss_valid))

        early_stopping(loss_valid, [f1, f2])
        if early_stopping.early_stop:
            print("Early stopping on epoch {}".format(epoch))
            break
        epoch += 1

    f1.load_state_dict(torch.load('checkpoint_0.pt'))
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.float().cuda()
        y_batch = y_batch.float().cuda()

        output = f1(X_batch)
        loss_test = criterion(output, y_batch).item()
        logging.info("Test loss {}".format(loss_test))
    return loss_test


kfold_split = response_data.kfold_patient_split(5)
data_loaders = []

for split in kfold_split:
    max_drug_indices_train = split.X_train[drug_col] == 1
    max_drug_indices_valid = split.X_valid[drug_col] == 1
    max_drug_indices_test = split.X_test[drug_col] == 1

    split.X_train = split.X_train[max_drug_indices_train].groupby("patient_id").mean()
    split.X_test = split.X_test[max_drug_indices_test].groupby("patient_id").mean()
    split.X_valid = split.X_valid[max_drug_indices_valid].groupby("patient_id").mean()
    split.y_train = split.y_train[max_drug_indices_train].groupby("patient_id").mean()
    split.y_test = split.y_test[max_drug_indices_test].groupby("patient_id").mean()
    split.y_valid = split.y_valid[max_drug_indices_valid].groupby("patient_id").mean()

    split.X_train.drop(response_data.drug_columns, axis=1, inplace=True)
    split.X_test.drop(response_data.drug_columns, axis=1, inplace=True)
    split.X_valid.drop(response_data.drug_columns, axis=1, inplace=True)

    X_train = split.X_train
    y_train = split.y_train

    X_test = split.X_test
    y_test = split.y_test

    X_valid = split.X_valid
    y_valid = split.y_valid

    feature_scaler = StandardScaler()
    outcome_scaler = StandardScaler()

    logging.info("Fitting feature scalers")
    feature_scaler.fit(X_train[X_train.columns])
    outcome_scaler.fit(y_train[y_train.columns])
    logging.info("Finished fitting feature scalers")

    logging.info("Applying feature scalar")
    X_train[X_train.columns] = feature_scaler.transform(X_train[X_train.columns])
    X_test[X_test.columns] = feature_scaler.transform(X_test[X_test.columns])
    X_valid[X_valid.columns] = feature_scaler.transform(X_valid[X_valid.columns])

    y_train[y_train.columns] = outcome_scaler.transform(y_train[y_train.columns])
    y_test[y_test.columns] = outcome_scaler.transform(y_test[y_test.columns])
    y_valid[y_valid.columns] = outcome_scaler.transform(y_valid[y_valid.columns])
    logging.info("Finished scaling features")

    # Create random Tensors to hold inputs and outputs
    train_dataset = data.ExVivoDrugData(X_train, y_train)
    test_dataset = data.ExVivoDrugData(X_test, y_test)
    valid_dataset = data.ExVivoDrugData(X_valid, y_valid)

    batch_size = 32

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size)
    test_loader = DataLoader(dataset=test_dataset, batch_size=X_test.shape[0])
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=X_valid.shape[0])

    data_loaders.append((train_loader, test_loader, valid_loader))

test_errors_no_prior = []
APExp = egexplainer.VariableBatchExplainer(train_dataset)

learning_rate_no_prior = 1e-3
learning_rate_with_prior = 1e-3
prior_learning_rate = 5e-5

"""
test_errors_merge_algo = []
prior_info = merge_data.data.T
prior_info = torch.FloatTensor(prior_info.values).cuda()
for train_loader, test_loader, valid_loader in data_loaders:
    D_in, D_out = X_train.shape[1], 1
    f1 = LinearModel(D_in=D_in, D_out=D_out).cuda().float()
    f1_optimizer = optim.Adam(f1.parameters(), lr=5e-5)

    f2 = LinearModel(
        D_in=prior_info.shape[1],
        D_out=1,
    ).cuda().float()

    f2_optimizer = optim.Adam(f2.parameters(), lr=5e-5)

    test_errors_merge_algo.append(train_with_merge_algo(f1, f2, f1_optimizer, f2_optimizer, prior_info))

aml_results_dir = "./aml_results/"
if not os.path.exists(aml_results_dir):
    os.makedirs(aml_results_dir)

pickle.dump(test_errors_merge_algo, open("{}/test_errors_merge_algo.p".format(aml_results_dir), "wb"))

for train_loader, test_loader, valid_loader in data_loaders:
    D_in, H1, H2, D_out = train_loader.dataset.X.shape[1], 512, 256, 1
    f1_no_prior = MLP(D_in=D_in, H1=H1, H2=H2, D_out=D_out, dropout=0).cuda().float()
    f1_no_prior_optimizer = optim.Adam(f1_no_prior.parameters(), lr=learning_rate_no_prior)

    test_errors_no_prior.append(train(f1_no_prior, f1_no_prior_optimizer))


test_errors_with_prior = []
prior_info = merge_data.data.T
prior_info = torch.FloatTensor(prior_info.values).cuda()
prior_mlp_errors = []

for train_loader, test_loader, valid_loader in data_loaders:
    D_in, H1, H2, D_out = X_train.shape[1], 512, 256, 1
    f1 = MLP(D_in=D_in, H1=H1, H2=H2, D_out=D_out, dropout=0.5).cuda().float()
    f1_optimizer = optim.Adam(f1.parameters(), lr=learning_rate_with_prior)

    f2 = MLP(
        D_in=prior_info.shape[1],
        H1=5,
        H2=3,
        D_out=1,
        dropout=0.0
    ).cuda().float()

    f2_optimizer = optim.Adam(f2.parameters(), lr=prior_learning_rate)

    test_errors_with_prior.append(train_with_learned_prior(f1, f2, f1_optimizer, f2_optimizer, prior_info))

test_errors_random_prior = []
genes = merge_data.data.T.index
mu = 0
sigma = 1

logging.info("Beginning training with random prior...")

prior_info = pd.DataFrame(np.random.normal(mu, sigma, merge_data.data.T.shape))
prior_info = prior_info.set_index(genes)
prior_info = torch.FloatTensor(prior_info.values).cuda()

for train_loader, test_loader, valid_loader in data_loaders:
    D_in, H1, H2, D_out = X_train.shape[1], 512, 256, 1
    f1_random_prior = MLP(D_in=D_in, H1=H1, H2=H2, D_out=D_out, dropout=0).cuda().float()
    f1_random_prior_optimizer = optim.Adam(f1_random_prior.parameters(), lr=learning_rate_with_prior)

    f2_random_prior = MLP(
        D_in=prior_info.shape[1],
        H1=5,
        H2=3,
        D_out=1,
        dropout=0.0
    ).cuda().float()

    f2_random_prior_optimizer = optim.Adam(f2_random_prior.parameters(), lr=prior_learning_rate)

    test_errors_random_prior.append(train_with_learned_prior(
        f1_random_prior, f2_random_prior, f1_random_prior_optimizer, f2_random_prior_optimizer, prior_info))

"""
logging.info("Training LASSO baseline...")
from sklearn import linear_model

potential_alphas = np.linspace(start=0, stop=1, num=100)[1:]
lasso_errors = []

best_mse_sum = float("inf")
best_alpha = None
for alpha in tqdm(potential_alphas):
    mse_sum = 0
    for train_loader, test_loader, valid_loader in data_loaders:
        clf = linear_model.Lasso(alpha=alpha)
        clf.fit(train_loader.dataset.X, train_loader.dataset.y)
        predictions = clf.predict(valid_loader.dataset.X)
        mse = ((predictions - valid_loader.dataset.y.values)**2).mean()
        mse_sum += mse
    if mse_sum < best_mse_sum:
        best_alpha = alpha
        best_mse_sum = mse_sum

logging.info("Best alpha: {}".format(best_alpha))
exit(1)
test_lasso_errors = []

for train_loader, test_loader, valid_loader in data_loaders:
    clf = linear_model.Lasso(alpha=best_alpha)
    clf.fit(train_loader.dataset.X, train_loader.dataset.y)
    predictions = clf.predict(test_loader.dataset.X)
    mse = ((predictions - test_loader.dataset.y.values) ** 2).mean()
    test_lasso_errors.append(mse)

print("Test lasso errors: {}".format(test_lasso_errors))
print("Test errors without prior: {}".format(test_errors_no_prior))
print("Test errors with random prior: {}".format(test_errors_random_prior))
print("Test errors with MERGE prior: {}".format(test_errors_with_prior))

aml_results_dir = "./aml_results/{}".format(args.drug)
if not os.path.exists(aml_results_dir):
    os.makedirs(aml_results_dir)

pickle.dump(test_errors_no_prior, open("{}/test_errors_no_prior.p".format(aml_results_dir), "wb"))
pickle.dump(test_errors_with_prior, open("{}/test_errors_with_prior.p".format(aml_results_dir), "wb"))
pickle.dump(test_errors_random_prior, open("{}/test_errors_random_prior.p".format(aml_results_dir), "wb"))
pickle.dump(test_lasso_errors, open("{}/test_lasso_errors.p".format(aml_results_dir), "wb"))

aml_models_dir = "./aml_models/{}".format(args.drug)
if not os.path.exists(aml_models_dir):
    os.makedirs(aml_models_dir)

#torch.save(f1_no_prior, "{}/prediction_model_no_prior.pth".format(aml_models_dir))
torch.save(f1, "{}/prediction_model_with_prior.pth".format(aml_models_dir))
torch.save(f2, "{}/prior.pth".format(aml_models_dir))

