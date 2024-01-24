import torch
import torch.nn as nn
import json

from src.data_manipulation.transformers.audio_spectograms.signal_to_freq_time_analysis import \
    transform_mix_and_bass_to_spectrogram

with open("./config/hyperparameters_audio.json", "r") as f:
    hyperparameters = json.load(f)

# relative paths to dataset as seen from this main.py file
BASE_PATH = "./data/raw/V1"
TRAIN_FOLDER_PATH = "./data/processed/train"
TRAIN_FILE_NAME = "mix_bass_train_data"
TRAIN_FILE_PATH = f"{TRAIN_FOLDER_PATH}/{TRAIN_FILE_NAME}.npz"


def main():
    transform_mix_and_bass_to_spectrogram(base_path=BASE_PATH, train_file_path=TRAIN_FILE_PATH)


if __name__ == "__main__":
    main()

# model = RNN(input_size=dict_size, output_size=dict_size, hidden_dim=hyperparameters['hidden_dim'], n_layers=hyperparameters['n_layers'])
