import torch
import numpy as np
from models import MLP, LinearModel
from torch import optim
from utils import train, train_with_learned_prior
from torch.nn import CrossEntropyLoss
import egexplainer
import pickle
import os
import logging
from scipy.stats import sem
from utils import configure_logging
from data import generate_two_moons_data

configure_logging()

nuisance_dim_values = [x for x in range(50, 1000+1, 50)]

learning_rate = 5e-4
prior_learning_rate = 1e-3
dropout = 0.0

patience = 100
num_experiments = 5

no_prior_mean_losses = []
with_prior_mean_losses = []
no_prior_loss_std = []
with_prior_loss_std = []

for num_nuisance_dims in nuisance_dim_values:
    losses_no_prior = []
    losses_with_prior = []
    logging.info("Running experiments with {} nuisance dimensions".format(num_nuisance_dims))

    for i in range(0, num_experiments):
        train_loader, valid_loader, test_loader = generate_two_moons_data(n_noisy_dimensions=num_nuisance_dims)

        D_in = train_loader.dataset.X.shape[1]
        H1 = D_in // 4
        H2 = H1 // 2
        D_out = 2

        f1 = MLP(D_in=D_in, H1=H1, H2=H2, D_out=D_out, dropout=dropout).cuda()

        f1_optimizer = optim.Adam(f1.parameters(), lr=learning_rate, weight_decay=5e-3)
        losses_no_prior.append(
            train(f1, f1_optimizer, CrossEntropyLoss(), train_loader, valid_loader, test_loader, patience=100)
        )

        D_in = train_loader.dataset.X.shape[1]
        H1 = D_in // 4
        H2 = H1 // 2
        D_out = 2
        f1 = MLP(D_in=D_in, H1=H1, H2=H2, D_out=D_out, dropout=dropout).cuda()

        f1_optimizer = optim.Adam(f1.parameters(), lr=learning_rate, weight_decay=5e-3)

        std = train_loader.dataset.X.std(axis=0).reshape(-1, 1)
        mean = train_loader.dataset.X.mean(axis=0).reshape(-1, 1)
        prior_info = np.concatenate((std, mean), axis=1)
        prior_info = torch.FloatTensor(prior_info).cuda()

        f2 = LinearModel(D_in=prior_info.shape[1], D_out=1).cuda()
        f2_optimizer = optim.Adam(f2.parameters(), lr=prior_learning_rate)
        APExp = egexplainer.VariableBatchExplainer(train_loader.dataset)

        losses_with_prior.append(
            train_with_learned_prior(
                f1, f2, f1_optimizer, f2_optimizer, CrossEntropyLoss(), train_loader, valid_loader,
                test_loader, patience, APExp, prior_info
            )
        )

    losses_no_prior = np.array(losses_no_prior)
    losses_with_prior = np.array(losses_with_prior)

    no_prior_mean_losses.append(losses_no_prior.mean())
    with_prior_mean_losses.append(losses_with_prior.mean())

    no_prior_loss_std.append(sem(losses_no_prior))
    with_prior_loss_std.append(sem(losses_with_prior))

two_moons_results_dir = "./two_moons_results"
if not os.path.exists(two_moons_results_dir):
    os.mkdir(two_moons_results_dir)

pickle.dump(no_prior_mean_losses, open("{}/no_prior_mean_losses.p".format(two_moons_results_dir), "wb"))
pickle.dump(with_prior_mean_losses, open("{}/with_prior_mean_losses.p".format(two_moons_results_dir), "wb"))
pickle.dump(no_prior_loss_std, open("{}/no_prior_loss_std.p".format(two_moons_results_dir), "wb"))
pickle.dump(with_prior_loss_std, open("{}/with_prior_loss_std.p".format(two_moons_results_dir), "wb"))
