import torch
import torch.nn as nn
import json


with open("../../../config/hyperparameters_audio.json") as hyperparameters_file:
    hyperparameters = json.load(hyperparameters_file)


class RNN(nn.Module):
    def __init__(self, input_size, hidden_dim, n_layers, output_size):
        super(RNN, self).__init__()

        # Number of hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.n_layers = n_layers

        # RNN
        self.rnn = nn.RNN(
            input_size, hidden_dim, n_layers, batch_first=True, nonlinearity="relu"
        )

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        # x = tensor of shape (batch_size, seq_length, input_size)

        batch_size = hyperparameters["batch_size"]

        # Initialize hidden state for first input
        hidden_i = self.init_hidden(batch_size)

        # Pass in input and hidden state and obtain outputs
        out, hidden_j = self.rnn(x, hidden_i)

        # Reshapes the tensor to have self.hidden_dim columns and the according number of rows to keep the same total size
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out[:, -1, :])

        return out, hidden_j

    def init_hidden(self, batch_size):
        # Generates the first hidden state of zeros for the forward pass
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)

        return hidden
