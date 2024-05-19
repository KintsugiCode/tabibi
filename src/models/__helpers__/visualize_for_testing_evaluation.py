from src.config.constants import VISUALIZATION_SAVE_PATH
from src.visualization.spectrograms.visualize_spectrogram import visualize_spectrogram


def visualize_for_testing_evaluation(y_pred, y_test, x_test, data, tag):
    # Convert tensor back into numpy array
    y_pred_for_visualization = y_pred.detach().cpu().numpy()
    x_test_for_visualization = x_test.detach().cpu().numpy()
    y_test_for_visualization = y_test.detach().cpu().numpy()

    visualize_spectrogram(
        save_folder_path=VISUALIZATION_SAVE_PATH,
        mel_spectrogram=x_test_for_visualization[0],
        mix_name=data["mix_name"],
        tag=tag,
        label="x_test",
    )
    visualize_spectrogram(
        save_folder_path=VISUALIZATION_SAVE_PATH,
        mel_spectrogram=y_pred_for_visualization[0],
        mix_name=data["mix_name"],
        tag=tag,
        label="y_pred",
    )
    visualize_spectrogram(
        save_folder_path=VISUALIZATION_SAVE_PATH,
        mel_spectrogram=y_test_for_visualization[0],
        mix_name=data["mix_name"],
        tag=tag,
        label="y_test",
    )
