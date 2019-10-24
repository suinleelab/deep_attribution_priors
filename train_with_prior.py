from data import ExVivoDrugData, MergeData
import logging
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
    "--lambda_start",
    help="Starting coefficient for sequence of prior lambdas",
    type=float
)
parser.add_argument(
    "--lambda_end",
    help="Final coefficient for sequence of prior lambdas",
    type=float
)
parser.add_argument(
    "--num_lambdas",
    help="How many lambda values to test (between the start + final)",
    type=int
)
parser.add_argument(
    "--include_baseline",
    help="Whether to include an additional MLP baseline (i.e., lambda=0)",
    type=bool
)

args = parser.parse_args()

response_data = ExVivoDrugData.load_data()
merge_data = MergeData.load_data()

response_data.ensure_overlap_with_prior(merge_data)
merge_data.ensure_overlap_with_response(response_data)

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

    prior_feature = merge_data.get_feature(args.prior_feature)
    prior_feature = torch.FloatTensor(prior_feature).cuda().reshape(-1,)

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
    APExp = AttributionPriorExplainer(background_dataset, N, k=2)

    attribution_prior = StaticFeatureAttributionPrior(
        explainer=APExp, prior_feature=prior_feature, ignored_features=response_data.drug_columns)
    epochs = 60

    prior_alphas = np.linspace(
        start=args.alpha_start,
        stop=args.alpha_end,
        num=args.alpha_lambdas
    )

    model = MLP(D_in=D_in, H1=H1, H2=H2, D_out=D_out, dropout=0.2).cuda().float()

    train(0.0, model, epochs, train_loader, test_loader, attribution_prior)
    exit(0)
    #lambdas_and_losses = []

    #if args.include_baseline:
    #lambdas_and_losses.append((0.0, ))
    #exit(0)

    #for alpha in prior_alphas:
    #    lambdas_and_losses.append((alpha, train(alpha, model, epochs, )))

    #for (lambda_value, loss) in lambdas_and_losses:
    #    print("{} : {}".format(lambda_value, loss))

    #with open("results.p", "wb") as handle:
    #    pickle.dump(lambdas_and_losses, handle, protocol=pickle.HIGHEST_PROTOCOL)