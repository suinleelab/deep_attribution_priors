import torch
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import os
import logging
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from data import ADData
import egexplainer
from models import MLP, ShallowMLP
from torch import optim
import time
import datetime
from utils import EarlyStopping
from tqdm import tqdm
import pickle
from IPython.core.debugger import set_trace

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

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

##
## put everything in 0-1 by region
##

# Read all region expressions
data_dir = "/projects/leelab2/jjanizek/"
regions = ['ACT_FWM', 'ACT_HIP', 'ACT_PCx', 'ACT_TCx', 'MSBB_BM10', 'MSBB_BM22', 'MSBB_BM36', 'MSBB_BM44', 'ROSMAP']

data_list = []

logging.info("Loading ABETA values")
for region in regions:
    # Read expression
    data_df = pd.read_table(data_dir + 'AD_DATA/abeta_' + region + '.csv', index_col=0, sep=',')

    data_df.columns = ['abeta']

    #standardized_df = (data_df - data_df.min()) / (data_df.max() - data_df.min())

    # Append to list
    data_list.append(data_df)

    # Join all data
joined_df = pd.concat(data_list, axis=0)
y = joined_df

logging.info("Loading gene expression data")
X = pd.read_csv(data_dir + 'AD_DATA/AD_DATA/AD_RnaSeq_Expression_Standardized_Batch_Corrected.tsv',sep='\t',index_col=0)

cram_features = pd.read_csv("./cram.csv")
overlapping_genes = list(set(X.columns).intersection(cram_features.columns))
X = X[overlapping_genes]
cram_features = cram_features[overlapping_genes]
cram_features = cram_features.T
prior_info = torch.FloatTensor(cram_features.values).cuda()

n_splits = 5
kf = KFold(n_splits=n_splits)
data_loaders = []

for train_idxs, test_idxs in kf.split(X):
    # We use a 60-20-20 train-valid-test split.  We first split the data into 80-20 train-test, then
    # get the validation fold from the training data
    train_idxs, val_idxs = train_test_split(train_idxs, test_size=0.25)
    X_train, X_val, X_test = X.iloc[train_idxs], X.iloc[val_idxs], X.iloc[test_idxs]
    y_train, y_val, y_test = y.iloc[train_idxs], y.iloc[val_idxs], y.iloc[test_idxs]

    y_train = y_train.values.reshape(-1, 1)
    y_test = y_test.values.reshape(-1, 1)
    y_val = y_val.values.reshape(-1, 1)

    feature_scaler = StandardScaler()
    outcome_scaler = StandardScaler()

    feature_scaler.fit(X_train)
    outcome_scaler.fit(y_train)

    X_train_ss, X_val_ss, X_test_ss = feature_scaler.transform(X_train), feature_scaler.transform(
        X_val), feature_scaler.transform(X_test)
    y_train_ss, y_val_ss, y_test_ss = outcome_scaler.transform(y_train), outcome_scaler.transform(
        y_val), outcome_scaler.transform(y_test)

    N = 32
    train_dataset = ADData(X_train_ss, y_train_ss)
    val_dataset = ADData(X_val_ss, y_val_ss)
    test_dataset = ADData(X_test_ss, y_test_ss)

    train_loader = DataLoader(dataset=train_dataset, batch_size=N)
    valid_loader = DataLoader(dataset=val_dataset, batch_size=N)
    test_loader = DataLoader(dataset=test_dataset, batch_size=test_dataset.X.shape[0])

    data_loaders.append((train_loader, test_loader, valid_loader))

APExp = egexplainer.VariableBatchExplainer(train_dataset)
criterion = torch.nn.MSELoss()

learning_rate = 1e-5
prior_learning_rate = 1e-4
patience = 100

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

        if epoch % 10 == 0:
            logging.info("Epoch {} completed in {} secs with valid loss {:.4f}".format(epoch, epoch_time, loss_valid))

        early_stopping(loss_valid, model)

        if early_stopping.early_stop:
            print("Early stopping on epoch {}".format(epoch))
            print("Final valid loss {}".format(early_stopping.val_metric_min))
            break
        epoch += 1
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.float().cuda()
        y_batch = y_batch.float().cuda()

        output = model(X_batch)
        loss_test = criterion(output, y_batch).item()
    return loss_test

test_errors_no_prior = []
for train_loader, test_loader, valid_loader in data_loaders:
    D_in, H1, H2, D_out = train_loader.dataset.X.shape[1], 512, 256, 1
    f1_no_prior = MLP(D_in=D_in, H1=H1, H2=H2, D_out=D_out, dropout=0).cuda().float()
    f1_no_prior_optimizer = optim.Adam(f1_no_prior.parameters(), lr=learning_rate)

    test_errors_no_prior.append(train(f1_no_prior, f1_no_prior_optimizer))

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
            prior_loss = ((prior_differences - eg) ** 2).mean()

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
            print("Final valid loss {}".format(loss_valid))
            break
        epoch += 1
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.float().cuda()
        y_batch = y_batch.float().cuda()

        output = f1(X_batch)
        loss_test = criterion(output, y_batch).item()
    return loss_test


test_errors_with_prior = []
for train_loader, test_loader, valid_loader in data_loaders:
    D_in, H1, H2, D_out = X_train.shape[1], 512, 256, 1

    f1 = MLP(D_in=D_in, H1=H1, H2=H2, D_out=D_out, dropout=0).cuda().float()
    f1_optimizer = torch.optim.Adam(f1.parameters(), lr=learning_rate)

    f2 = ShallowMLP(
        D_in=prior_info.shape[1],
        H1=4,
        D_out=1,
        dropout=0.5
    ).cuda().float()
    f2_optimizer = torch.optim.Adam(f2.parameters(), lr=prior_learning_rate)

    test_errors_with_prior.append(train_with_learned_prior(f1, f2, f1_optimizer, f2_optimizer, prior_info))

test_errors_random_prior = []
mu = 0
sigma = 1

logging.info("Beginning training with random prior...")

prior_info = pd.DataFrame(np.random.normal(mu, sigma, cram_features.shape))
prior_info = torch.FloatTensor(prior_info.values).cuda()

for train_loader, test_loader, valid_loader in data_loaders:
    D_in, H1, H2, D_out = X_train.shape[1], 512, 256, 1
    f1_random_prior = MLP(D_in=D_in, H1=H1, H2=H2, D_out=D_out, dropout=0).cuda().float()
    f1_random_prior_optimizer = optim.Adam(f1_random_prior.parameters(), lr=learning_rate)

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
        mse = ((predictions - valid_loader.dataset.y)**2).mean()
        mse_sum += mse
    if mse_sum < best_mse_sum:
        best_alpha = alpha
        best_mse_sum = mse_sum

test_lasso_errors = []

for train_loader, test_loader, valid_loader in data_loaders:
    clf = linear_model.Lasso(alpha=best_alpha)
    clf.fit(train_loader.dataset.X, train_loader.dataset.y)
    predictions = clf.predict(test_loader.dataset.X)
    mse = ((predictions - test_loader.dataset.y) ** 2).mean()
    test_lasso_errors.append(mse)

print("Test lasso errors: {}".format(test_lasso_errors))
print("Test errors without prior: {}".format(test_errors_no_prior))
print("Test errors with random prior: {}".format(test_errors_random_prior))
print("Test errors with MERGE prior: {}".format(test_errors_with_prior))

ad_results_dir = "./ad_results/"
if not os.path.exists(ad_results_dir):
    os.makedirs(ad_results_dir)

pickle.dump(test_errors_no_prior, open("{}/test_errors_no_prior.p".format(ad_results_dir), "wb"))
pickle.dump(test_errors_with_prior, open("{}/test_errors_with_prior.p".format(ad_results_dir), "wb"))
pickle.dump(test_errors_random_prior, open("{}/test_errors_random_prior.p".format(ad_results_dir), "wb"))
pickle.dump(test_lasso_errors, open("{}/test_lasso_errors.p".format(ad_results_dir), "wb"))

ad_models_dir = "./ad_models/"
if not os.path.exists(ad_models_dir):
    os.makedirs(ad_models_dir)

#torch.save(f1_no_prior, "{}/prediction_model_no_prior.pth".format(aml_models_dir))
torch.save(f1, "{}/prediction_model_with_prior.pth".format(ad_models_dir))
torch.save(f2, "{}/prior.pth".format(ad_models_dir))



