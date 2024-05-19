from src.models.__helpers__.visualize_for_testing_evaluation import (
    visualize_for_testing_evaluation,
)


def test(x_test, y_test, model, criterion, data, tag):
    print("@@@@ Starting model testing @@@@")
    # Switch the model to evaluation mode to turn off features like dropout
    model.eval()

    # Pass x_test through the model to get y_pred
    y_pred, _ = model(x_test)

    test_loss = criterion(y_pred, y_test)
    print(f"@@@@ Test loss: {test_loss.item():.7f} @@@@")

    visualize_for_testing_evaluation(y_pred, y_test, x_test, data, tag)

    # Convert tensor back into numpy array
    y_pred = y_pred.detach().cpu().numpy()

    return y_pred
