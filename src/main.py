import torch
import torch.nn as nn
import json


with open("src/config/hyperparameters.json", "r") as f:
    hyperparameters = json.load(f)


# model = RNN(input_size=dict_size, output_size=dict_size, hidden_dim=hyperparameters['hidden_dim'], n_layers=hyperparameters['n_layers'])
