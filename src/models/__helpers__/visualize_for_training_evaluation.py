from src.config.constants import VISUALIZATION_SAVE_PATH, TRAINED_AUDIO_FILE_PATH
from src.transformers.freq_time_analysis_to_audio import freq_time_analysis_to_audio
from src.visualization.spectrograms.visualize_spectrogram import visualize_spectrogram


def visualize_for_training_evaluation(outputs, x, y, data, flag, tag):
    # Convert tensor back into numpy array
    outputs_for_visualization = outputs.detach().cpu().numpy()
    x_train_for_visualization = x.detach().cpu().numpy()
    y_train_for_visualization = y.detach().cpu().numpy()

    visualize_spectrogram(
        save_folder_path=VISUALIZATION_SAVE_PATH,
        mel_spectrogram=x_train_for_visualization[0],
        mix_name=data["mix_name"],
        tag=tag,
        label="x_train",
    )
    visualize_spectrogram(
        save_folder_path=VISUALIZATION_SAVE_PATH,
        mel_spectrogram=y_train_for_visualization[0],
        mix_name=data["mix_name"],
        tag=tag,
        label="y_train",
    )
    visualize_spectrogram(
        save_folder_path=VISUALIZATION_SAVE_PATH,
        mel_spectrogram=outputs_for_visualization[0],
        mix_name=data["mix_name"],
        tag=tag,
        label="y_output",
    )

    if flag:
        freq_time_analysis_to_audio(
            mel_spectrogram_array=outputs_for_visualization[:3],
            output_file_path=TRAINED_AUDIO_FILE_PATH,
            mix_names=data["mix_name"],
            min_max_amplitudes=data["y_min_max_amplitudes"],
            tag=tag,
        )
