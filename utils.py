import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset

def make_mlp(input_dim, hidden_dims, output_dim=1, activation='relu'):
    """Create a mlp from the configurations.
    """
    activation = {
        'relu': nn.ReLU
    }[activation]

    num_hidden_layers = len(hidden_dims)

    seq = [nn.Linear(input_dim, hidden_dims[0]), activation()]
    for i in range(num_hidden_layers-1):
        seq += [nn.Linear(hidden_dims[i], hidden_dims[i+1]), activation()]
    seq += [nn.Linear(hidden_dims[-1], output_dim)]

    return nn.Sequential(*seq)

def load_and_cache_examples(x_samples_path, y_samples_path):
    x_samples = np.load(x_samples_path)
    y_samples = np.load(y_samples_path)

    gaussian_dataset = GaussianSampleDataset(x_samples,
                                             y_samples)

    return gaussian_dataset

class GaussianSampleDataset(Dataset):

    def __init__(self, x_gaussian_samples, y_gaussian_samples):
        self.x_gaussian_samples = x_gaussian_samples
        self.y_gaussian_samples = y_gaussian_samples

    def __len__(self):
        return np.shape(self.x_gaussian_samples)[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x_sample = self.x_gaussian_samples[idx, :]
        x_sample = torch.tensor(x_sample, dtype=torch.float32)
        y_sample = self.y_gaussian_samples[idx, :]
        y_sample = torch.tensor(y_sample, dtype=torch.float32)

        sample = [x_sample, y_sample]

        return sample

def permute_samples(x_samples, y_samples):
    """ The permutation function that shuffles the two sets of samples
    Returns:
        (permuted x samples, 
        permuted y samples)
    """
    device = x_samples.device
    batch_size = x_samples.size(0)
    feature_length = x_samples.size(1)

    x_samples = torch.reshape(x_samples, (batch_size, 1, feature_length))
    y_samples = torch.reshape(y_samples, (batch_size, 1, feature_length))

    for i in range(batch_size):
        j = i+1 if i < batch_size - 1 else 0

        permutation_label = torch.zeros([1,1,1], dtype=torch.float32, device=device)

        x_sample_p = torch.reshape(x_samples[i], (1, 1, feature_length))
        y_sample_p = torch.reshape(y_samples[j], (1, 1, feature_length))

        if i == 0:
            permuted_x_samples = x_sample_p
            permuted_y_samples = y_sample_p
        else:
            permuted_x_samples = torch.cat((permuted_x_samples, x_sample_p), 0)
            permuted_y_samples = torch.cat((permuted_y_samples, y_sample_p), 0)

    return permuted_x_samples, permuted_y_samples

def cat_samples(x_samples, y_samples, permuted_x_samples, permuted_y_samples):
    """ The concatenation function that concatenates the original samples and permuted samples
    Returns:
        (new x samples, 
        new y samples, 
        the matching flags between them)
    """
    device = x_samples.device
    batch_size_original = x_samples.size(0)
    batch_size_permuted = permuted_x_samples.size(0)
    feature_length = x_samples.size(1)

    x_samples = torch.reshape(x_samples, (batch_size_original, 1, feature_length))
    y_samples = torch.reshape(y_samples, (batch_size_original, 1, feature_length))

    new_x_samples = x_samples
    new_y_samples = y_samples
    matching_flags = torch.ones([batch_size_original, 1, 1], dtype=torch.float32, device=device)
    permutation_labels = torch.zeros([batch_size_permuted,1,1], dtype=torch.float32, device=device)

    new_x_samples = torch.cat((new_x_samples, permuted_x_samples), 0)
    new_y_samples = torch.cat((new_y_samples, permuted_y_samples), 0)
    matching_flags = torch.cat((matching_flags, permutation_labels), 0)

    return new_x_samples, new_y_samples, matching_flags

def reshape_samples(x_samples, y_samples):
    """ The function that reshapes two sets of samples 
    Returns:
        (reshaped x samples, 
        reshaped y samples)
    """
    device = x_samples.device
    batch_size = x_samples.size(0)
    feature_length = x_samples.size(1)

    x_samples = torch.reshape(x_samples, (batch_size, 1, feature_length))
    y_samples = torch.reshape(y_samples, (batch_size, 1, feature_length))

    return x_samples, y_samples