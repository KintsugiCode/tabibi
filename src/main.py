import torch
import torch.nn as nn
import json
from src.__helpers__.__utils__ import load_numpy_data
from src.data_manipulation.data_splitter.train_test_split import (
    train_test_splitter,
)
from src.data_manipulation.transformers.audio_spectrograms.signal_to_freq_time_analysis import (
    transform_mix_and_bass_to_spectrogram,
)
from src.data_manipulation.transformers.truncating.mix_bass_data_truncator import (
    data_overall_truncator,
)
from src.models.audio_separation.gru.gru import GRU
from src.transformers.freq_time_analysis_to_audio import freq_time_analysis_to_audio

# relative paths to dataset as seen from this main.py file
subset = "V1"
BASE_PATH = f"./data/raw/{subset}"

TRAIN_FOLDER_PATH = "./data/processed/train"
TRAIN_FILE_NAME = f"mix_bass_train_data_{subset}TEST"

TEST_FOLDER_PATH = "./data/processed/test"
TEST_FILE_NAME = f"mix_bass_test_data_{subset}TEST"


TRAIN_FILE_PATH = f"{TRAIN_FOLDER_PATH}/normalized_{TRAIN_FILE_NAME}.npz"
TEST_FILE_PATH = f"{TEST_FOLDER_PATH}/normalized_{TEST_FILE_NAME}.npz"

TRAINED_AUDIO_FILE_PATH = "./visualization/trained_audio"
PRED_AUDIO_FILE_PATH = "./visualization/predicted_audio"


with open("./config/hyperparameters_audio.json") as hyperparameters_file:
    hyperparameters = json.load(hyperparameters_file)


def main():
    # Split data into train/test
    print("@@@@@@ Splitting data into train/test @@@@@@")
    train_files, test_files = train_test_splitter(BASE_PATH)

    # Transform training data and receive max_dimension
    print("@@@@@@ Transforming training data @@@@@@")

    (min_dimension_train) = transform_mix_and_bass_to_spectrogram(
        base_path=BASE_PATH,
        files_to_transform=train_files,
        save_file_path=TRAIN_FILE_PATH,
    )
    # Transform testing data and receive max_dimension
    print("@@@@@@ Transforming testing data @@@@@@")
    (min_dimension_test) = transform_mix_and_bass_to_spectrogram(
        base_path=BASE_PATH,
        files_to_transform=test_files,
        save_file_path=TEST_FILE_PATH,
    )

    # Load the training dataset
    print("@@@@@@ Loading the training dataset @@@@@@")
    data_train = load_numpy_data(
        f"{TRAIN_FOLDER_PATH}/normalized_{TRAIN_FILE_NAME}.npz"
    )

    # Truncate the training dataset to the smallest dimension
    data_train = data_overall_truncator(
        data_train, min_dimension_train, min_dimension_test
    )

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
        model.parameters(), lr=hyperparameters["learning_rate"]
    )

    # Track loss to break training loop if loss is no longer changing
    no_change = 0
    prev_loss = float("inf")

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

        if epoch == hyperparameters["n_epochs"] - 1:
            # Convert tensor back into numpy array and then back to audio
            outputs_for_visualization = outputs.detach().cpu().numpy()
            # Convert first three tracks back to audio for review
            freq_time_analysis_to_audio(
                outputs_for_visualization[:3],
                TRAINED_AUDIO_FILE_PATH,
                data_train["mix_name"],
                data_train["min_max_amplitudes"],
                flag="TRAINING-",
            )
        print("@@@@@@ Calculating loss @@@@@@")
        loss = criterion(outputs, y_train)

        print("@@@@@@ Running backward pass and updating model parameters @@@@@@")
        # Backward pass (backpropagation) where gradients are calculated
        loss.backward()
        # Update model parameters, based on the gradients calculated in the backward pass
        optimizer.step()

        # print statistics
        print(f"@@@@@@ Epoch {epoch + 1} Done. loss: {loss.item():.7f} @@@@@@")

        # Check if loss is not changing
        if abs(prev_loss - loss.item()) < 1e-7:  # small threshold to count as no change
            no_change += 1
        else:
            no_change = 0

        prev_loss = loss.item()

        if no_change >= 20:
            print("@@@@@@ Stopping early - loss hasn't changed in 20 epochs. @@@@@@ ")
            break

    # Load the testing dataset
    print("@@@@@@ Loading the testing dataset @@@@@@")
    data_test = load_numpy_data(f"{TEST_FOLDER_PATH}/normalized_{TEST_FILE_NAME}.npz")

    # Truncate the training dataset to the smallest dimension
    data_test = data_overall_truncator(
        data_test, min_dimension_train, min_dimension_test
    )

    # Convert to PyTorch Tensor -- Individual conversion before grouped conversion is faster for large datasets
    print("@@@@@@ Converting to PyTorch Tensor @@@@@@")
    x_test = torch.stack([torch.tensor(x) for x in data_test["x"]])
    y_test = torch.stack([torch.tensor(y) for y in data_test["y"]])

    # Test the model
    print("@@@@@@ Starting model testing @@@@@@")
    # Switch the model to evaluation mode to turn off features like dropout
    model.eval()

    # Pass x_test through the model to get y_pred
    y_pred, _ = model(x_test)

    test_loss = criterion(y_pred, y_test)
    print(f"Test loss: {test_loss.item():.7f}")

    # Convert tensor back into numpy array and then back to audio
    y_pred = y_pred.detach().cpu().numpy()
    # Convert first three tracks back to audio for review
    freq_time_analysis_to_audio(
        y_pred[:3],
        PRED_AUDIO_FILE_PATH,
        data_test["mix_name"],
        data_test["min_max_amplitudes"],
        flag="TESTING-",
    )


if __name__ == "__main__":
    main()
