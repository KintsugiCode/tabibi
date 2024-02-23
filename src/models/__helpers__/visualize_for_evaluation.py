from src.__helpers__.constants import VISUALIZATION_SAVE_PATH, TRAINED_AUDIO_FILE_PATH
from src.transformers.freq_time_analysis_to_audio import freq_time_analysis_to_audio
from src.visualization.spectrograms.visualize_spectrograms import visualize_spectrograms


def visualize_for_evaluation(outputs, x, y, data, flag, tag):
    # Convert tensor back into numpy array and then back to audio
    outputs_for_visualization = outputs.detach().cpu().numpy()
    # Convert first three tracks back to audio for review
    x_train_for_visualization = x.detach().cpu().numpy()
    y_train_for_visualization = y.detach().cpu().numpy()
    visualize_spectrograms(
        VISUALIZATION_SAVE_PATH,
        x_train_for_visualization[0],
        y_train_for_visualization[0],
        outputs_for_visualization[0],
        data["mix_name"],
        tag=tag,
    )
    if flag:
        freq_time_analysis_to_audio(
            outputs_for_visualization[:3],
            data["y_phase"],
            TRAINED_AUDIO_FILE_PATH,
            data["mix_name"],
            data["min_max_amplitudes"],
            tag=tag,
        )
