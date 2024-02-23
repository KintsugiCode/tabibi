import json
import os

from src.data_manipulation.normalization.mix_bass_data_normalizer import (
    Normalizer,
)
from src.transformers.audio_to_freq_time_analysis import audio_to_freq_time_analysis
from src.transformers.freq_time_analysis_to_audio import freq_time_analysis_to_audio


os.chdir(os.path.dirname(os.path.abspath(__file__)))


with open("../../config/fourierparameters.json") as fourierparameters_file:
    fourierparameters = json.load(fourierparameters_file)

INPUT_FILE_PATH = "audio/MusicDelta_80sRock/Bass/MusicDelta_80sRock_STEM_02.wav"


def fourier_audio_loss(file_path):
    try:
        mel_spectrogram, phase = audio_to_freq_time_analysis(file_path)
        t_dict = {"x": list(), "phase": list(), "min_max_amplitudes": list()}

        t_dict["x"].append(mel_spectrogram)

        norm_x = Normalizer(t_dict["x"])
        t_dict["x"], t_dict["min_max_amplitudes"] = (
            norm_x.normalize(),
            norm_x.get_min_max(),
        )
        t_dict["phase"].append(phase)

        freq_time_analysis_to_audio(
            t_dict["x"],
            t_dict["phase"],
            "audio/",
            ["MusicDelta_80sRock_MIX.wav"],
            t_dict["min_max_amplitudes"],
            flag="TRIAL-",
        )

    except Exception as e:
        raise Exception("Exception occurred: {}".format(e))


fourier_audio_loss(INPUT_FILE_PATH)
