import json
import os
import torch
from torch import nn

from src.__helpers__.__utils__ import load_numpy_data
from src.__helpers__.constants import (
    MODEL2_TRAIN_FOLDER_PATH,
    MODEL2_TRAIN_FILE_NAME,
    MODEL2_TEST_FOLDER_PATH,
    MODEL2_TEST_FILE_NAME,
    TRAINED_MODEL2_SAVE_PATH,
)
from src.data_manipulation.transform_data import transform_data
from src.data_manipulation.truncator.mix_bass_data_truncator import (
    data_truncator,
)
from src.models.tab_transcription.gru.gru_transcription import GRU_Transcription
from src.models.test import test
from src.models.train import train

dir_path = os.path.dirname(os.path.realpath(__file__))
hyperparameters_path = os.path.join(
    dir_path, "../../config/hyperparameters_transcription.json"
)

with open(hyperparameters_path) as hyperparameters_file:
    hyperparameters = json.load(hyperparameters_file)


def transcription_manager():
    # Transform training/testing data for tab transcription
    transform_data(flag="tab transcription")

    # Load the training dataset
    print("@@@@@@ Loading the training dataset @@@@@@")
    data_train = load_numpy_data(
        f"{MODEL2_TRAIN_FOLDER_PATH}/normalized_{MODEL2_TRAIN_FILE_NAME}.npz"
    )

    # Load the testing dataset
    print("@@@@@@ Loading the testing dataset @@@@@@")
    data_test = load_numpy_data(
        f"{MODEL2_TEST_FOLDER_PATH}/normalized_{MODEL2_TEST_FILE_NAME}.npz"
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
    print("@@@@@@ Converting to PyTorch Tensor @@@@@@")
    x_train = torch.stack([torch.tensor(x) for x in data_train["x"]])
    y_train = torch.stack([torch.tensor(y) for y in data_train["y"]])

    x_train = x_train.float()
    y_train = y_train.float()

    # Convert to PyTorch Tensor -- Individual conversion before grouped conversion is faster for large datasets
    print("@@@@@@ Converting to PyTorch Tensor @@@@@@")
    x_test = torch.stack([torch.tensor(x) for x in data_test["x"]])
    y_test = torch.stack([torch.tensor(y) for y in data_test["y"]])

    x_test = x_test.float()
    y_test = y_test.float()

    # Initialize the model
    print("@@@@@@ Initializing the model @@@@@@")
    model = GRU_Transcription(
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
        x_train, y_train, model, criterion, optimizer, data_train, tag="transcription"
    )

    # Save the trained model
    print("@@@@@@ Saving trained model @@@@@@")
    torch.save(model.state_dict(), TRAINED_MODEL2_SAVE_PATH)

    # Test the model
    y_pred = test(x_test, y_test, model, criterion)
