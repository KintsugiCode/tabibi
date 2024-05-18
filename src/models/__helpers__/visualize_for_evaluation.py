from src.config.constants import VISUALIZATION_SAVE_PATH, TRAINED_AUDIO_FILE_PATH
from src.data_manipulation.__helpers__.normalization.decibel_normalizer import (
    DecibelNormalizer,
)
from src.transformers.freq_time_analysis_to_audio import freq_time_analysis_to_audio
from src.visualization.spectrograms.visualize_spectrograms import visualize_spectrograms


def visualize_for_evaluation(outputs, x, y, data, flag, tag):
    # Convert tensor back into numpy array
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
            mel_spectrogram_array=outputs_for_visualization[:3],
            output_file_path=TRAINED_AUDIO_FILE_PATH,
            mix_names=data["mix_name"],
            min_max_amplitudes=data["y_min_max_amplitudes"],
            tag=tag,
        )
