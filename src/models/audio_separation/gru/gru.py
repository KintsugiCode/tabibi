from torch import nn
import torch


class GRU(nn.Module):  # Changed class name to GRU
    def __init__(self, input_size, hidden_dim, n_layers, output_size, dropout_rate):
        super(GRU, self).__init__()  # Changed to GRU

        # Number of hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.n_layers = n_layers

        # Dropout rate
        self.dropout_rate = dropout_rate

        # GRU
        self.gru = nn.GRU(input_size, hidden_dim, n_layers, batch_first=True)

        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

        # Readout layers - Increased capacity with the addition of another fully connected layer
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim * 4)
        self.fc3 = nn.Linear(hidden_dim * 4, output_size)

    def forward(self, x):
        batch_size = x.size(0)

        # Initialize hidden state for first input
        hidden_i = self.init_hidden(batch_size)

        # Pass in input and hidden state and obtain outputs
        out, hidden_j = self.gru(x, hidden_i)  # Changed to gru

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
