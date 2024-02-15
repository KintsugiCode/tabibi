import torch
import torch.nn as nn
import json
import os
from src.__helpers__.__utils__ import load_numpy_data
from src.__helpers__.constants import (
    TRAINED_MODEL_SAVE_PATH,
    PRED_AUDIO_FILE_PATH,
    MODEL1_TRAIN_FOLDER_PATH,
    MODEL1_TRAIN_FILE_NAME,
    MODEL1_TEST_FOLDER_PATH,
    MODEL1_TEST_FILE_NAME,
)
from src.data_manipulation.transformers.transform_data import transform_data
from src.data_manipulation.transformers.truncating.mix_bass_data_truncator import (
    data_overall_truncator,
)
from src.models.audio_separation.gru.gru import GRU
from src.models.audio_separation.test_separation import test_separation
from src.models.audio_separation.train_separation import train_separation
from src.transformers.freq_time_analysis_to_audio import freq_time_analysis_to_audio

dir_path = os.path.dirname(os.path.realpath(__file__))
hyperparameters_path = os.path.join(dir_path, "./config/hyperparameters_audio.json")

with open(hyperparameters_path) as hyperparameters_file:
    hyperparameters = json.load(hyperparameters_file)


def main():
    # Transform training/testing data
    transform_data(flag="audio separation")

    # Load the training dataset
    print("@@@@@@ Loading the training dataset @@@@@@")
    data_train = load_numpy_data(
        f"{MODEL1_TRAIN_FOLDER_PATH}/normalized_{MODEL1_TRAIN_FILE_NAME}.npz"
    )

    # Load the testing dataset
    print("@@@@@@ Loading the testing dataset @@@@@@")
    data_test = load_numpy_data(
        f"{MODEL1_TEST_FOLDER_PATH}/normalized_{MODEL1_TEST_FILE_NAME}.npz"
    )

    # Truncate the training dataset to the smallest dimension
    data_train = data_overall_truncator(
        data_train, data_train["min_dimension"], data_test["min_dimension"]
    )

    # Truncate the testing dataset to the smallest dimension
    data_test = data_overall_truncator(
        data_test, data_train["min_dimension"], data_test["min_dimension"]
    )

    # Begin training
    # Convert to PyTorch Tensor -- Individual conversion before grouped conversion is faster for large datasets
    print("@@@@@@ Converting to PyTorch Tensor @@@@@@")
    x_train = torch.stack([torch.tensor(x) for x in data_train["x"]])
    y_train = torch.stack([torch.tensor(y) for y in data_train["y"]])

    # Initialize the model
    print("@@@@@@ Initializing the model @@@@@@")
    model = GRU(
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
    model = train_separation(x_train, y_train, model, criterion, optimizer, data_train)

    # Save the trained model
    print("@@@@@@ Saving trained model @@@@@@")
    torch.save(model.state_dict(), TRAINED_MODEL_SAVE_PATH)

    # Begin testing
    # Convert to PyTorch Tensor -- Individual conversion before grouped conversion is faster for large datasets
    print("@@@@@@ Converting to PyTorch Tensor @@@@@@")
    x_test = torch.stack([torch.tensor(x) for x in data_test["x"]])
    y_test = torch.stack([torch.tensor(y) for y in data_test["y"]])

    # Test the model
    y_pred = test_separation(x_test, y_test, model, criterion)

    # Convert first three tracks back to audio for review
    freq_time_analysis_to_audio(
        y_pred[:3],
        data_test["y_phase"],
        PRED_AUDIO_FILE_PATH,
        data_test["mix_name"],
        data_test["min_max_amplitudes"],
        flag="TESTING",
    )


if __name__ == "__main__":
    main()
