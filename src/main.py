import json
import numpy as np
from sklearn.decomposition import NMF
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.io import wavfile
import librosa

with open("src/config/hyperparameters.json", "r") as f:
    hyperparameters = json.load(f)


def main():
    # call data transformer with audio filepath

    # call the rnn-model with the prepared data
        # model = RNN(input_size=dict_size, output_size=dict_size, hidden_dim=hyperparameters['hidden_dim'], n_layers=hyperparameters['n_layers'])

    # visualize the evaluation metrics that the model returns


if __name__ == "__main__":
    main()





