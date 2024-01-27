import torch
import torch.nn as nn
import json


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
        """
        `x` is the batch of sequences that you want your RNN to process.

        `x` has a shape of `(batch_size, seq_length, num_features)`:
            - `batch_size` is the number of sequences you process at a time.
            - `seq_length` is the length of each sequence.
            - `num_features` is the number of input features at each sequence element.
        """
        batch_size = x.size(0)

        # Initialize hidden state for first input
        hidden_i = self.init_hidden(batch_size)

        # Pass in input and hidden state and obtain outputsinput_size: Number of features of your input vector
        #         hidden_size: Number of hidden neurons
        #         output_size: Number of features of your output vector
        out, hidden_j = self.rnn(x, hidden_i)

        out = self.fc(out)

        return out, hidden_j

    def init_hidden(self, batch_size):
        # Generates the first hidden state of zeros for the forward pass
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)

        return hidden
