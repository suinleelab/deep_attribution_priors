import logging
from prior import StaticFeatureAttributionPrior
from torch.utils.data import DataLoader
import numpy as np
import torch.optim as optim


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

    logging.info("Training model using\n{}\nfor prior penalty".format(attribution_prior.prior_feature))
    for epoch in range(epochs):
        model.train()

        train_losses = []
        valid_losses = []

        # Train
        for i, (features, labels) in enumerate(training_data):
            features, labels = features.cuda().float(), labels.cuda().float()
            optimizer.zero_grad()
            outputs = model(features)

            loss = model.criterion(outputs, labels) + attribution_prior.penalty(model, features)

            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        for i, (features, labels) in enumerate(test_data):
            features, labels = features.cuda().float(), labels.cuda().float()
            outputs = model(features)
            valid_losses.append(model.criterion(outputs, labels).detach().cpu().numpy())

        valid_loss = np.mean(valid_losses)

        logging.info("{}, {}, {}".format(epoch, np.mean(train_losses), valid_loss))

    return valid_loss
