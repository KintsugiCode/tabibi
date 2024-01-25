import torch
import torch.nn as nn
import json

from src.data_manipulation.data_splitter.train_test_split import (
    train_test_splitter,
)
from src.data_manipulation.transformers.audio_spectograms.signal_to_freq_time_analysis import (
    transform_mix_and_bass_to_spectrogram,
)

with open("./config/hyperparameters_audio.json", "r") as f:
    hyperparameters = json.load(f)

subset = "V2"

# relative paths to dataset as seen from this main.py file
BASE_PATH = f"./data/raw/{subset}"

TRAIN_FOLDER_PATH = "./data/processed/train"
TRAIN_FILE_NAME = f"mix_bass_train_data_{subset}"

TEST_FOLDER_PATH = "./data/processed/test"
TEST_FILE_NAME = f"mix_bass_test_data_{subset}"


TRAIN_FILE_PATH = f"{TRAIN_FOLDER_PATH}/{TRAIN_FILE_NAME}.npz"
TEST_FILE_PATH = f"{TEST_FOLDER_PATH}/{TEST_FILE_NAME}.npz"


def main():
    train_files, test_files = train_test_splitter(BASE_PATH)

    # Transform training data
    transform_mix_and_bass_to_spectrogram(
        base_path=BASE_PATH,
        files_to_transform=train_files,
        save_file_path=TRAIN_FILE_PATH,
    )
    # Transform testing data
    transform_mix_and_bass_to_spectrogram(
        base_path=BASE_PATH,
        files_to_transform=test_files,
        save_file_path=TEST_FILE_PATH,
    )


if __name__ == "__main__":
    main()

# model = RNN(input_size=dict_size, output_size=dict_size, hidden_dim=hyperparameters['hidden_dim'], n_layers=hyperparameters['n_layers'])
