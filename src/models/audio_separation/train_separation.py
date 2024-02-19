import json
import os

from src.models.__helpers__.learning_rate_reducer import learning_rate_reducer
from src.models.__helpers__.visualize_for_evaluation import visualize_for_evaluation

dir_path = os.path.dirname(os.path.realpath(__file__))
hyperparameters_path = os.path.join(dir_path, "../../config/hyperparameters_audio.json")

with open(hyperparameters_path) as hyperparameters_file:
    hyperparameters = json.load(hyperparameters_file)


def train_separation(x_train, y_train, model, criterion, optimizer, data_train):
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

    return model
