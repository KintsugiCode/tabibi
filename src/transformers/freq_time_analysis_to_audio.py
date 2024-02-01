import librosa
import soundfile as sf

from src.data_manipulation.transformers.normalization.mix_bass_data_normalizer import (
    Normalizer,
)


def freq_time_analysis_to_audio(
    spectrogram_array, output_file_path, mix_names, min_max_amplitudes, flag
):
    try:
        for track in range(spectrogram_array.shape[0]):
            min_val, max_val = min_max_amplitudes
            spectrogram_array[track] = Normalizer(spectrogram_array[track]).denormalize(
                min_val, max_val
            )
            print(f"@@@@@@ Recreating audio of track {track} @@@@@@")
            # inverts stft whilst estimating phase -- is much clearer than librosa.istft
            audio = librosa.griffinlim(spectrogram_array[track])
            # Save the audio to file
            sf.write(f"{output_file_path}/{flag}{mix_names[track]}", audio, 22050)
        print("@@@@@@ Audio recreation completed @@@@@@")

    except Exception as e:
        raise Exception("Exception occurred: {}".format(e))
