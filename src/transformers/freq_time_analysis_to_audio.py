import math

import librosa
import soundfile as sf
import json
import os

from src.data_manipulation.normalization.mix_bass_data_normalizer import (
    Normalizer,
)
from scipy.signal import butter, lfilter

os.chdir(os.path.dirname(os.path.abspath(__file__)))

with open("../config/fourierparameters.json") as fourierparameters_file:
    fourierparameters = json.load(fourierparameters_file)


def butter_lowpass(cutoff, sr, order=5):
    nyq = 0.5 * sr
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


def lowpass_filter(data, cutoff, sr, order=5):
    b, a = butter_lowpass(cutoff, sr, order=order)
    filtered_data = lfilter(b, a, data)
    return filtered_data


def freq_time_analysis_to_audio(
    mel_spectrogram_array,
    phase_array,
    output_file_path,
    mix_names,
    min_max_amplitudes,
    tag,
):
    try:
        track_counter = 0

        for track in range(mel_spectrogram_array.shape[0]):
            track_counter += 1
            min_val, max_val = min_max_amplitudes
            mel_spectrogram_array[track] = Normalizer(
                mel_spectrogram_array[track]
            ).denormalize(min_val, max_val)
            print(f"@@@@@@ Recreating audio of track {track} @@@@@@")
            spectrogram_array = librosa.feature.inverse.mel_to_stft(
                mel_spectrogram_array,
                sr=fourierparameters["sample_rate"],
                n_fft=fourierparameters["n_fft"],
                power=1.0,
            )

            # Rounded-up sample length of track from audio to freq-time analysis conversion
            length = math.ceil(
                (
                    fourierparameters["sample_rate"]
                    * fourierparameters["track_seconds_considered"]
                )
                / fourierparameters["hop_length"]
            )

            # inverts stft whilst estimating phase -- is much clearer than librosa.istft
            audio = librosa.griffinlim(
                spectrogram_array[track],
                hop_length=fourierparameters["hop_length"],
                win_length=fourierparameters["n_fft"],
                n_fft=fourierparameters["n_fft"],
            )

            audio = lowpass_filter(
                audio, cutoff=1500, sr=fourierparameters["sample_rate"]
            )

            # Save the audio to file
            sf.write(
                f"{output_file_path}/{track_counter}-{tag}-{mix_names[track]}",
                audio,
                fourierparameters["sample_rate"],
            )
        print("@@@@@@ Audio recreation completed @@@@@@")

    except Exception as e:
        raise Exception("Exception occurred: {}".format(e))
