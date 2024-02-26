import numpy as np
from src.__helpers__.__utils__ import (
    convert_t_dict_key_to_numpy_arrays,
    convert_to_recarray,
)
from src.data_manipulation.__helpers__.normalization.mix_bass_data_normalizer import (
    Normalizer,
)
from src.transformers.audio_to_freq_time_analysis import audio_to_freq_time_analysis


def production_audio_to_dict(file_name, input_path):
    t_dict = {
        "x": list(),
        "y": list(),
        "mix_name": list(),
    }

    try:
        if file_name.endswith(".wav"):

            mix_spectrogram, _ = audio_to_freq_time_analysis(file_path=input_path)

            t_dict["mix_name"].append(file_name)
            t_dict["x"].append(mix_spectrogram)

            # Append an array of zeroes to y to initialize with matching dimensions to x
            zero_spectrogram = np.zeros_like(mix_spectrogram)
            t_dict["y"].append(zero_spectrogram)

    except Exception as e:
        raise Exception("An error occurred in processing {}: {}".format(file_name, e))

    try:
        # Transform to array
        t_dict = convert_t_dict_key_to_numpy_arrays(dictionary=t_dict, keys=["x", "y"])

        # Normalize the data
        norm_x = Normalizer(t_dict["x"])
        t_dict["x"], t_dict["min_max_amplitudes"] = (
            norm_x.normalize(),
            norm_x.get_min_max(),
        )

    except Exception as e:
        raise Exception(
            "An error occurred during spectrogram-to-data-format conversion."
        )

    return t_dict
