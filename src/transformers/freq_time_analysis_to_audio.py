import librosa
import soundfile as sf

from src.data_manipulation.transformers.normalization.mix_bass_data_normalizer import (
    Normalizer,
)


def freq_time_analysis_to_audio(
    mel_spectrogram_array, output_file_path, mix_names, min_max_amplitudes, flag
):
    try:
        for track in range(mel_spectrogram_array.shape[0]):
            min_val, max_val = min_max_amplitudes
            mel_spectrogram_array[track] = Normalizer(
                mel_spectrogram_array[track]
            ).denormalize(min_val, max_val)
            print(f"@@@@@@ Recreating audio of track {track} @@@@@@")
            spectrogram_array = librosa.feature.inverse.mel_to_stft(
                mel_spectrogram_array, sr=44100, n_fft=2048, power=1.0
            )
            # inverts stft whilst estimating phase -- is much clearer than librosa.istft
            audio = librosa.griffinlim(spectrogram_array[track])
            # Save the audio to file
            sf.write(f"{output_file_path}/{flag}{mix_names[track]}", audio, 22050)
        print("@@@@@@ Audio recreation completed @@@@@@")

    except Exception as e:
        raise Exception("Exception occurred: {}".format(e))
