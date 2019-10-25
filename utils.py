import logging
from prior import StaticFeatureAttributionPrior
from torch.utils.data import DataLoader
import numpy as np
import torch.optim as optim
from egexplainer import ExpectedGradientsModel


def train(model,
          epochs: int,
          training_data: DataLoader,
          test_data: DataLoader,
          attribution_prior: StaticFeatureAttributionPrior):
    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-5,
        weight_decay=1e-4
    )

    if attribution_prior is not None:
        logging.info("Training model using\n{}\nfor prior penalty".format(attribution_prior.prior_feature))
    else:
        logging.info("Training model with no attribution prior")
    for epoch in range(epochs):
        model.train()

        train_model_errors = []
        train_prior_penalties = []
        valid_losses = []

        # Train
        for i, (features, labels) in enumerate(training_data):
            features, labels = features.cuda().float(), labels.cuda().float()
            optimizer.zero_grad()

            #pdb.set_trace()

            if attribution_prior is not None:
                outputs, shaps = model(features)
                model_error = model.base.criterion(outputs, labels)
                prior_penalty = attribution_prior.penalty(shaps)
                train_prior_penalties.append(prior_penalty.mean().item())
                loss = model_error + prior_penalty
            else:
                outputs = model(features)
                model_error = model.criterion(outputs, labels)
                loss = model_error
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            train_model_errors.append(model_error.mean().item())

        # Validation
        for i, (features, labels) in enumerate(test_data):
            features, labels = features.cuda().float(), labels.cuda().float()
            if isinstance(model, ExpectedGradientsModel):
                outputs, shaps = model(features)
                valid_losses.append(model.base.criterion(outputs, labels).mean().detach().cpu().numpy())
            else:
                outputs = model(features)
                valid_losses.append(model.criterion(outputs, labels).mean().detach().cpu().numpy())


        valid_loss = np.mean(valid_losses)
        if attribution_prior is not None:
            logging.info("Epoch {}, Training Model Error {}, Training Prior Penalty {}, Validation Error {}".format(
                epoch, np.mean(train_model_errors), np.mean(train_prior_penalties), valid_loss))
        else:
            logging.info("{}, {}, {}".format(epoch, np.mean(train_model_errors), valid_loss))

    return valid_loss
