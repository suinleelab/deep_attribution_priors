from sklearn import datasets
from utils import BasicDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import numpy as np

class BasicDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        sample = self.X[index]
        sample_label = self.y[index]
        return sample, sample_label


def generate_two_moons_data(n_noisy_dimensions):
    n_samples = 1000
    batch_size = 32
    X, y = datasets.make_moons(n_samples=n_samples, noise=0.1)
    X = (X).astype(np.float32)

    noise_mean = 0
    noise_var = 1.0
    noise = np.random.normal(
        loc=noise_mean,
        scale=noise_var,
        size=(n_samples, n_noisy_dimensions)
    )

    X_with_noise = np.concatenate((X, noise), axis=1)
    X_with_noise = X_with_noise.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X_with_noise, y, test_size=0.8
    )

    X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5)

    train_dataset = BasicDataset(X_train, y_train)
    valid_dataset = BasicDataset(X_valid, y_valid)
    test_dataset = BasicDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    valid_loader = DataLoader(valid_dataset, batch_size=y_valid.shape[0])
    test_loader = DataLoader(test_dataset, batch_size=y_test.shape[0])

    return train_loader, valid_loader, test_loader
