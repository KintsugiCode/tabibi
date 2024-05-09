import json
import os
import torch
from torch import nn
from src.__helpers__.__utils__ import load_numpy_data
from src.config.constants import (
    MODEL1_TRAIN_FOLDER_PATH,
    MODEL1_TRAIN_FILE_NAME,
    MODEL1_TEST_FOLDER_PATH,
    MODEL1_TEST_FILE_NAME,
    PRED_AUDIO_FILE_PATH,
    TRAINED_MODEL1_SAVE_PATH,
)
from src.data_manipulation.transform_data import transform_data
from src.data_manipulation.__helpers__.truncator.mix_bass_data_truncator import (
    data_truncator,
)
from src.models.audio_separation.gru.gru_separation import GRU_Separation
from src.models.test import test
from src.models.train import train
from src.transformers.freq_time_analysis_to_audio import freq_time_analysis_to_audio

dir_path = os.path.dirname(os.path.realpath(__file__))
hyperparameters_path = os.path.join(
    dir_path, "../../config/hyperparameters_separation.json"
)

with open(hyperparameters_path) as hyperparameters_file:
    hyperparameters = json.load(hyperparameters_file)


def separation_manager():
    while True:
        choice = (
            input(
                "@@@@@@ Would you like to pre-process the data [1] or use the existing pre-processed data [2]?:"
            )
            .strip()
            .lower()
        )

        if choice in ["1"]:
            # Transform training/testing data for audio separation
            print()
            print("@@@@@@ DATA PRE-PROCESSING START @@@@@@")
            transform_data(flag="audio separation")

            break

        elif choice in ["2"]:
            break

        else:
            print("Please enter a valid input. Either [1] or [2].")

    # Load the training dataset
    print("@@@@ Loading the training dataset @@@@")
    data_train = load_numpy_data(
        f"{MODEL1_TRAIN_FOLDER_PATH}/normalized_{MODEL1_TRAIN_FILE_NAME}.npz"
    )

    # Load the testing dataset
    print("@@@@ Loading the testing dataset @@@@")
    data_test = load_numpy_data(
        f"{MODEL1_TEST_FOLDER_PATH}/normalized_{MODEL1_TEST_FILE_NAME}.npz"
    )

    # Truncate the training dataset to the smallest dimension
    data_train = data_truncator(
        data=data_train,
        min_dimension_train=data_train["min_dimension"],
        min_dimension_test=data_test["min_dimension"],
        flag="overall",
    )

    # Truncate the testing dataset to the smallest dimension
    data_test = data_truncator(
        data=data_test,
        min_dimension_train=data_train["min_dimension"],
        min_dimension_test=data_test["min_dimension"],
        flag="overall",
    )

    # Convert to PyTorch Tensor -- Individual conversion before grouped conversion is faster for large datasets
    x_train = torch.stack([torch.tensor(x) for x in data_train["x"]])
    y_train = torch.stack([torch.tensor(y) for y in data_train["y"]])

    x_train = x_train.float()
    y_train = y_train.float()

    # Convert to PyTorch Tensor -- Individual conversion before grouped conversion is faster for large datasets
    x_test = torch.stack([torch.tensor(x) for x in data_test["x"]])
    y_test = torch.stack([torch.tensor(y) for y in data_test["y"]])

    x_test = x_test.float()
    y_test = y_test.float()

    # Initialize the model
    model = GRU_Separation(
        input_size=x_train.shape[2],
        hidden_dim=hyperparameters["hidden_dim"],
        n_layers=hyperparameters["n_layers"],
        output_size=y_train.shape[2],
        dropout_rate=hyperparameters["dropout_rate"],
    )

    # Choose a loss function and an optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=hyperparameters["learning_rate"], weight_decay=1e-6
    )

    # Train the model
    model = train(
        x_train, y_train, model, criterion, optimizer, data_train, tag="separation"
    )

    # Save the trained model
    print("@@@@ Saving trained model @@@@")
    torch.save(model.state_dict(), TRAINED_MODEL1_SAVE_PATH)

    # Test the model
    y_pred = test(x_test, y_test, model, criterion)

    # Convert first three tracks back to audio for review
    freq_time_analysis_to_audio(
        mel_spectrogram_array=y_pred[:3],
        output_file_path=PRED_AUDIO_FILE_PATH,
        mix_names=data_test["mix_name"],
        min_max_amplitudes=data_test["min_max_amplitudes"],
        tag="SEPARATION-TESTING",
    )
