import torch
import torch.nn as nn
import json
import os
from src.__helpers__.__utils__ import load_numpy_data
from src.__helpers__.constants import (
    TRAIN_FOLDER_PATH,
    TRAIN_FILE_NAME,
    TEST_FOLDER_PATH,
    TEST_FILE_NAME,
    VISUALIZATION_SAVE_PATH,
    TRAINED_AUDIO_FILE_PATH,
    TRAINED_MODEL_SAVE_PATH,
    PRED_AUDIO_FILE_PATH,
)
from src.data_manipulation.transformers.transform_data import transform_data
from src.data_manipulation.transformers.truncating.mix_bass_data_truncator import (
    data_overall_truncator,
)
from src.models.__helpers__.learning_rate_reducer import learning_rate_reducer
from src.models.__helpers__.visualize_for_evaluation import visualize_for_evaluation
from src.models.audio_separation.gru.gru import GRU
from src.transformers.freq_time_analysis_to_audio import freq_time_analysis_to_audio
from src.visualization.spectrograms_visualized.visualize_mag_spectrograms import (
    visualize_spectrograms,
)

dir_path = os.path.dirname(os.path.realpath(__file__))
hyperparameters_path = os.path.join(dir_path, "./config/hyperparameters_audio.json")

with open(hyperparameters_path) as hyperparameters_file:
    hyperparameters = json.load(hyperparameters_file)


def main():
    """
    # Uncomment to transform training/testing data if data dictionary not yet created
    transform_data()
    """

    # Load the training dataset
    print("@@@@@@ Loading the training dataset @@@@@@")
    data_train = load_numpy_data(
        f"{TRAIN_FOLDER_PATH}/normalized_{TRAIN_FILE_NAME}.npz"
    )

    # Load the testing dataset
    print("@@@@@@ Loading the testing dataset @@@@@@")
    data_test = load_numpy_data(f"{TEST_FOLDER_PATH}/normalized_{TEST_FILE_NAME}.npz")

    # Truncate the training dataset to the smallest dimension
    data_train = data_overall_truncator(
        data_train, data_train["min_dimension"], data_test["min_dimension"]
    )

    # Truncate the testing dataset to the smallest dimension
    data_test = data_overall_truncator(
        data_test, data_train["min_dimension"], data_test["min_dimension"]
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
        model.parameters(), lr=hyperparameters["learning_rate"], weight_decay=1e-6
    )

    # Track loss to break training loop if loss is no longer changing
    no_change = 0
    prev_loss = float("inf")

    # Track learning rate reduction
    lr_reduced = [False] * 10

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

        outputs, _ = model(
            x_train,
        )

        loss = criterion(outputs, y_train)

        # Backward pass (backpropagation) where gradients are calculated
        loss.backward()
        # Update model parameters, based on the gradients calculated in the backward pass
        optimizer.step()

        # print statistics
        print(f"@@@@@@ Epoch {epoch + 1} Done. loss: {loss.item():.7f} @@@@@@")

        # Reduce learning rate if necessary
        lr_reduced, optimizer = learning_rate_reducer(loss, optimizer, lr_reduced)

        # Visualizations and audio-transforms for manual evaluation
        if epoch == hyperparameters["n_epochs"] - 1:
            visualize_for_evaluation(
                outputs, x_train, y_train, data_train, flag="TRAINING"
            )

        # Check if loss is not changing anymore
        if abs(prev_loss - loss.item()) < 1e-8:  # small threshold to count as no change
            no_change += 1
        else:
            no_change = 0

        prev_loss = loss.item()

        # If loss hasn't changed for 30 epochs, stop early
        if no_change >= 30:
            print("@@@@@@ Stopping early - loss hasn't changed in 30 epochs. @@@@@@ ")
            visualize_for_evaluation(
                outputs, x_train, y_train, data_train, flag="TRAINING"
            )
            break

    # Save the trained model
    print("@@@@@@ Saving trained model @@@@@@")
    torch.save(model.state_dict(), TRAINED_MODEL_SAVE_PATH)

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
        data_test["y_phase"],
        PRED_AUDIO_FILE_PATH,
        data_test["mix_name"],
        data_test["min_max_amplitudes"],
        flag="TESTING",
    )


if __name__ == "__main__":
    main()
