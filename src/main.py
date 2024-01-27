import numpy as np
import torch
import torch.nn as nn
import json

from src.__helpers__.__utils__ import load_numpy_data
from src.data_manipulation.data_splitter.train_test_split import (
    train_test_splitter,
)
from src.data_manipulation.transformers.audio_spectograms.signal_to_freq_time_analysis import (
    transform_mix_and_bass_to_spectrogram,
)
from src.models.audio_separation.rnn.rnn import RNN


# relative paths to dataset as seen from this main.py file
subset = "V1"
BASE_PATH = f"./data/raw/{subset}"

TRAIN_FOLDER_PATH = "./data/processed/train"
TRAIN_FILE_NAME = f"mix_bass_train_data_{subset}TEST"

TEST_FOLDER_PATH = "./data/processed/test"
TEST_FILE_NAME = f"mix_bass_test_data_{subset}TEST"


TRAIN_FILE_PATH = f"{TRAIN_FOLDER_PATH}/{TRAIN_FILE_NAME}.npz"
TEST_FILE_PATH = f"{TEST_FOLDER_PATH}/{TEST_FILE_NAME}.npz"


with open("./config/hyperparameters_audio.json") as hyperparameters_file:
    hyperparameters = json.load(hyperparameters_file)


def main():
    # Split data into train/test
    print("@@@@@@ Splitting data into train/test @@@@@@")
    train_files, test_files = train_test_splitter(BASE_PATH)

    # Transform training data
    print("@@@@@@ Transforming training data @@@@@@")
    transform_mix_and_bass_to_spectrogram(
        base_path=BASE_PATH,
        files_to_transform=train_files,
        save_file_path=TRAIN_FILE_PATH,
    )
    # Transform testing data
    print("@@@@@@ Transforming testing data @@@@@@")
    transform_mix_and_bass_to_spectrogram(
        base_path=BASE_PATH,
        files_to_transform=test_files,
        save_file_path=TEST_FILE_PATH,
    )

    # Load the dataset
    print("@@@@@@ Loading the dataset @@@@@@")
    data = load_numpy_data(f"{TRAIN_FOLDER_PATH}/{TRAIN_FILE_NAME}.npz")

    # Convert to PyTorch Tensor
    print("@@@@@@ Converting to PyTorch Tensor @@@@@@")
    x_train = torch.tensor(data["x_train"])
    y_train = torch.tensor(data["y_train"])

    # Initialize the model
    print("@@@@@@ Initializing the model @@@@@@")
    model = RNN(
        input_size=x_train.shape[2],
        hidden_dim=hyperparameters["hidden_dim"],
        n_layers=hyperparameters["n_layers"],
        output_size=y_train.shape[2],
    )

    # Choose a loss function and an optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model with the data
    print("@@@@@@ Starting model training @@@@@@")
    for epoch in range(
        hyperparameters["n_epochs"]
    ):  # loop over the dataset multiple times
        print(f"@@@@@@ Starting Epoch {epoch + 1} @@@@@@")

        # zero the parameter gradients
        optimizer.zero_grad()

        """
        Forward pass through the model to generate the output predictions and then
        calculate the loss between the model's predictions (outputs), and the actual target values (y_train)

        """
        print("@@@@@@ Running forward pass @@@@@@")
        outputs, _ = model(
            x_train,
        )
        print("@@@@@@ Calculating loss @@@@@@")
        loss = criterion(outputs, y_train)

        print("@@@@@@ Running backward pass and updating model parameters @@@@@@")
        # Backward pass (backpropagation) where gradients are calculated
        loss.backward()
        # Update model parameters, based on the gradients calculated in the backward pass
        optimizer.step()

        # print statistics
        print(f"@@@@@@ Epoch {epoch + 1} Done. loss: {loss.item():.3f} @@@@@@")


if __name__ == "__main__":
    main()
