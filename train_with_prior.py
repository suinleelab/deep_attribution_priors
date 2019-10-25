from data import ExVivoDrugData, MergeData
import logging
from egexplainer import ExpectedGradientsModel
from sklearn.preprocessing import StandardScaler
from data import ExVivoDrugData
from prior import StaticFeatureAttributionPrior
import argparse
import logging
from utils import train

from models import MLP
from torch.utils.data import DataLoader
import torch
import numpy as np

from egexplainer import AttributionPriorExplainer

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--prior_feature",
    help="Which merge feature to use as an attribution prior",
    type=str,
    required=True
)

parser.add_argument(
    "--epochs",
    help="Number of training epochs",
    type=int,
    required=True
)

parser.add_argument(
    "--dropout",
    help="Dropout rate for neural network models",
    type=float,
    default=0.2
)

args = parser.parse_args()

response_data = ExVivoDrugData.load_data()
merge_data = MergeData.load_data()

overlapping_genes = list(set(response_data.X.columns).intersection(merge_data.data.columns))
merge_data.data = merge_data.data[overlapping_genes]
response_data.X = response_data.X[["patient_id"] + response_data.drug_columns + overlapping_genes]

logging.info("Splitting data:")
for train_test_split in response_data.kfold_patient_split(5):
    X_train = train_test_split.X_train
    y_train = train_test_split.y_train

    X_test = train_test_split.X_test
    y_test = train_test_split.y_test

    feature_scaler = StandardScaler()
    outcome_scaler = StandardScaler()

    logging.info("Fitting feature scalers")
    feature_scaler.fit(X_train[X_train.columns])
    outcome_scaler.fit(y_train[y_train.columns])
    logging.info("Finished fitting feature scalers")

    logging.info("Applying feature scalar")
    X_train[X_train.columns] = feature_scaler.transform(X_train[X_train.columns])
    X_test[X_test.columns] = feature_scaler.transform(X_test[X_test.columns])

    y_train[y_train.columns] = outcome_scaler.transform(y_train[y_train.columns])
    y_test[y_test.columns] = outcome_scaler.transform(y_test[y_test.columns])
    logging.info("Finished scaling features")

    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    N, D_in, H1, H2, D_out = 32*2*2, X_train.shape[1], 512, 256, 1

    # Create random Tensors to hold inputs and outputs
    train_dataset = ExVivoDrugData(X_train, y_train)
    test_dataset = ExVivoDrugData(X_test, y_test)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=N,
        shuffle=True,
        drop_last=True
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=N,
        shuffle=False,
        drop_last=False
    )

    background_dataset = ExVivoDrugData(X_train, y_train)
    if args.prior_feature != "NONE":
        prior_feature = merge_data.get_feature(args.prior_feature)
        base_model = MLP(D_in=D_in, H1=H1, H2=H2, D_out=D_out, dropout=args.dropout).cuda().float()
        model = ExpectedGradientsModel(base_model=base_model, refset=background_dataset)
        attribution_prior = StaticFeatureAttributionPrior(
            explainer=AttributionPriorExplainer,
            prior_feature=prior_feature,
            ignored_features=response_data.drug_columns,
            background_dataset=background_dataset
        )
    else:
        model = MLP(D_in=D_in, H1=H1, H2=H2, D_out=D_out, dropout=args.dropout).cuda().float()
        attribution_prior = None

    epochs = 100
    train(model, epochs, train_loader, test_loader, attribution_prior)
