import json
import os
from src.models.__helpers__.learning_rate_reducer import learning_rate_reducer
from src.models.__helpers__.visualize_for_evaluation import visualize_for_evaluation

dir_path = os.path.dirname(os.path.realpath(__file__))
hyperparameters_separation_path = os.path.join(
    dir_path, "../config/hyperparameters_separation.json"
)
hyperparameters_transcription_path = os.path.join(
    dir_path, "../config/hyperparameters_transcription.json"
)
with open(hyperparameters_separation_path) as hyperparameters_file:
    hyperparameters_separation = json.load(hyperparameters_file)
with open(hyperparameters_transcription_path) as hyperparameters_file:
    hyperparameters_transcription = json.load(hyperparameters_file)


def train(x_train, y_train, model, criterion, optimizer, data_train, tag):
    if tag == "separation":
        hyperparameters = hyperparameters_separation
    elif tag == "transcription":
        hyperparameters = hyperparameters_transcription
    else:
        raise Exception("Incorrect tag")

    # Track loss to break training loop if loss is no longer changing
    no_change = 0
    prev_loss = float("inf")

    # Track learning rate reduction
    lr_reduced = [False] * 10

    # Train the model with the training data
    print("@@@@ Starting model training @@@@")
    for epoch in range(hyperparameters["n_epochs"]):
        print(f"@@ Starting Epoch {epoch + 1} @@")

        # Zero the parameter gradients
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

        # Print statistics
        print(f"@@ Epoch {epoch + 1} Done. loss: {loss.item():.7f} @@")

        # Reduce learning rate if necessary
        lr_reduced, optimizer = learning_rate_reducer(loss, optimizer, lr_reduced)

        # Visualizations and audio-transforms for manual evaluation
        if epoch == hyperparameters["n_epochs"] - 1:
            visualize_for_evaluation(
                outputs,
                x_train,
                y_train,
                data_train,
                tag=f"{tag}-TRAINING",
                flag=False,
            )

        # Check if loss is not changing anymore
        if abs(prev_loss - loss.item()) < 1e-8:
            no_change += 1
        else:
            no_change = 0

        prev_loss = loss.item()

        # If loss hasn't changed for 30 epochs, stop early
        if no_change >= 30:
            print("@@@@@@ Stopping early - loss hasn't changed in 30 epochs. @@@@@@ ")
            visualize_for_evaluation(
                outputs,
                x_train,
                y_train,
                data_train,
                tag=f"{tag}-TRAINING",
                flag=False,
            )
            break

    return model
