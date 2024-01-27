import museval
import numpy as np
import torch
import torch.nn as nn
import json
from torch.nn.utils.rnn import pad_sequence
from src.__helpers__.__utils__ import load_numpy_data
from src.data_manipulation.data_splitter.train_test_split import (
    train_test_splitter,
)
from src.data_manipulation.transformers.audio_spectograms.signal_to_freq_time_analysis import (
    transform_mix_and_bass_to_spectrogram,
)
from src.data_manipulation.transformers.padding.mix_bass_data_padder import data_padder
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

    # Transform training data and receive max_dimension
    print("@@@@@@ Transforming training data @@@@@@")
    (
        max_dimension_train,
        x_length_train,
        y_length_train,
    ) = transform_mix_and_bass_to_spectrogram(
        base_path=BASE_PATH,
        files_to_transform=train_files,
        save_file_path=TRAIN_FILE_PATH,
        flag="train",
    )
    # Transform testing data and receive max_dimension
    print("@@@@@@ Transforming testing data @@@@@@")
    (
        max_dimension_test,
        x_length_test,
        y_length_test,
    ) = transform_mix_and_bass_to_spectrogram(
        base_path=BASE_PATH,
        files_to_transform=test_files,
        save_file_path=TEST_FILE_PATH,
        flag="test",
    )

    # Load the training dataset
    print("@@@@@@ Loading the training dataset @@@@@@")
    data = load_numpy_data(f"{TRAIN_FOLDER_PATH}/{TRAIN_FILE_NAME}.npz")

    # Pad the training dataset to the largest dimension
    data, *_ = data_padder(data, max_dimension_train, max_dimension_test)

    # Convert to PyTorch Tensor -- Individual conversion before grouped conversion is faster for large datasets
    print("@@@@@@ Converting to PyTorch Tensor @@@@@@")
    x_train = torch.stack([torch.tensor(x) for x in data["x"]])
    y_train = torch.stack([torch.tensor(y) for y in data["y"]])

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

    # Train the model with the training data
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

    # Load the testing dataset
    print("@@@@@@ Loading the testing dataset @@@@@@")
    data = load_numpy_data(f"{TEST_FOLDER_PATH}/{TEST_FILE_NAME}.npz")

    # Pad the training dataset to the largest dimension
    data, *_ = data_padder(data, max_dimension_train, max_dimension_test)

    # Convert to PyTorch Tensor -- Individual conversion before grouped conversion is faster for large datasets
    print("@@@@@@ Converting to PyTorch Tensor @@@@@@")
    x_test = torch.stack([torch.tensor(x) for x in data["x"]])
    y_test = torch.stack([torch.tensor(y) for y in data["y"]])

    # Test the model
    print("@@@@@@ Starting model testing @@@@@@")
    # Switch the model to evaluation mode to turn off features like dropout
    model.eval()

    # Pass x_test through the model to get y_pred
    y_pred, _ = model(x_test)

    # Reshape tensors to [nsrc, nsample, nchan] format for Museval as it requires this format.
    # We consider nchan = number of frequency buckets,
    # nsrc = number of sources (1, as our data is mono audio), and nsample = number of time steps.
    y_test = y_test.permute([0, 2, 1])
    y_pred = y_pred.permute(0, 2, 1)

    # No need to track gradients in testing, can speed up computations
    with torch.no_grad():
        for i in range(y_test.shape[0]):
            true_sources = y_test[i].numpy()
            estimated_sources = y_pred[i].numpy()
            # compute all three metrics (SDR, SIR, SAR)
            scores = museval.evaluate(true_sources, estimated_sources)
            for score_name, score_value in scores.items():
                print(f"{score_name} = {np.mean(score_value):.3f}")

    test_loss = criterion(y_pred, y_test)
    print(f"Test loss: {test_loss.item():.3f}")


if __name__ == "__main__":
    main()
