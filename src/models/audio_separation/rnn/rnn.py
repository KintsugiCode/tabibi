import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_size, hidden_dim, n_layers, output_size, dropout_rate):
        super(RNN, self).__init__()

        # Number of hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.n_layers = n_layers

        # Dropout rate
        self.dropout_rate = dropout_rate

        # RNN
        self.rnn = nn.RNN(
            input_size, hidden_dim, n_layers, batch_first=True, nonlinearity="relu"
        )

        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

        # Readout layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        """
        `x` is the batch of sequences that you want your RNN to process.

        `x` has a shape of `(batch_size, freq_bins, time_steps)`
        """
        batch_size = x.size(0)

        # Initialize hidden state for first input
        hidden_i = self.init_hidden(batch_size)

        # Pass in input and hidden state and obtain outputs
        out, hidden_j = self.rnn(x, hidden_i)

        # Pass output through dropout layer
        out = self.dropout(out)

        out = self.fc1(out)
        out = torch.relu(out)
        out = self.fc2(out)
        out = torch.relu(out)
        out = self.fc3(out)

        return out, hidden_j

    def init_hidden(self, batch_size):
        # Generates the first hidden state of zeros for the forward pass
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)

        return hidden
