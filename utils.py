import torch.optim as optim
import numpy as np

def train(alpha, model, epochs, training_data, test_data, attribution_prior):
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    print("Training model with prior penalty {}".format(alpha))
    for epoch in range(epochs):
        model.train()

        train_losses = []
        valid_losses = []

        # Train
        for i, (features, labels) in enumerate(training_data):
            features, labels = features.cuda().float(), labels.cuda().float()
            optimizer.zero_grad()
            outputs = model(features)

            loss = model.criterion(outputs, labels) + attribution_prior.penalty(model, features, alpha)

            loss.backward(retain_graph=True)
            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        for i, (features, labels) in enumerate(test_data):
            features, labels = features.cuda().float(), labels.cuda().float()
            outputs = model(features)
            valid_losses.append(model.criterion(outputs, labels).detach().cpu().numpy())
            shap_values = attribution_prior.explainer.shap_values(model, features)
            eg_no_drugs = shap_values[:, len(attribution_prior.ignored_features):]
            shap_sums = np.abs(eg_no_drugs).sum()
            shap_stdevs = eg_no_drugs.stdev()

            for (shap_sum, shap_stdev) in zip(shap_sums, shap_stdevs):
                print("{}, {}".format(shap_sum, shap_stdev))

        valid_loss = np.mean(valid_losses)

        if epoch % 10 == 0:
            print(epoch, np.mean(train_losses), valid_loss)
    return valid_loss