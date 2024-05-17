import os
import time
from src.__helpers__.__utils__ import (
    convert_t_dict_key_to_numpy_arrays,
    get_one_file_with_extension,
    savez_numpy_data,
    convert_to_recarray,
)
from src.data_manipulation.__helpers__.normalization.decibel_normalizer import (
    DecibelNormalizer,
)
from src.data_manipulation.__helpers__.truncator.mix_bass_data_truncator import (
    data_truncator,
)
from src.transformers.audio_to_freq_time_analysis import audio_to_freq_time_analysis


def mixed_signal_to_dict(base_path, files_to_transform, save_file_path, pause=False):
    t_dict = {
        "x": list(),
        "y": list(),
        "x_phase": list(),
        "y_phase": list(),
        "mix_name": list(),
        "min_dimension": 0,
    }

    data_point_multitude = 0
    if pause:
        data_point_multitude = 1

    data_point_amount = 0
    dim = []

    """
    Iterates over all folders in base_path and checks if the folder is included in the files_to_transform list.
    If yes, transforms the mix .wav file and the bass .wav file into spectrograms.
    """
    for foldername in os.listdir(f"{base_path}"):
        if foldername in files_to_transform:
            for mix_file_name in os.listdir(f"{base_path}/{foldername}/"):
                try:
                    if mix_file_name.endswith(".wav"):
                        print(f"@@ data_point: {mix_file_name} @@ ")

                        mix_folder_path = f"{base_path}/{foldername}"
                        mix_file_path = f"{mix_folder_path}/{mix_file_name}"

                        print(mix_file_name)

                        bass_folder_path = f"{mix_folder_path}/Bass"
                        bass_file_name = get_one_file_with_extension(
                            directory_path=bass_folder_path, extension="wav"
                        )
                        print(bass_file_name)
                        # Ignore all mix files where matching bass file is missing
                        if bass_file_name is None:
                            print("@@ SKIPPED -- Bass track not available @@")
                            print()
                            break

                        print()

                        if pause:
                            if data_point_amount == (data_point_multitude * 10):
                                print("@@@@ Waiting for 10 seconds @@@@")
                                print()
                                data_point_multitude += 1
                                time.sleep(10)

                        bass_file_path = f"{bass_folder_path}/{bass_file_name}"

                        mix_spectrogram, mix_phase = audio_to_freq_time_analysis(
                            file_path=mix_file_path
                        )
                        bass_spectrogram, bass_phase = audio_to_freq_time_analysis(
                            file_path=bass_file_path
                        )

                        t_dict["x"].append(mix_spectrogram)
                        t_dict["y"].append(bass_spectrogram)
                        t_dict["x_phase"].append(mix_phase)
                        t_dict["y_phase"].append(bass_phase)
                        t_dict["mix_name"].append(mix_file_name)

                        # Track the dimensions for later padding/truncating
                        dim.append(mix_spectrogram.shape[1])

                        data_point_amount += 1
                except Exception as e:
                    raise Exception(
                        "An error occurred in processing {}: {}".format(foldername, e)
                    )

    try:
        # Save min_dimension to later truncate the dataset again after overall min_dimension of datasets is known
        min_dimension = min(dim)
    except Exception as e:
        raise Exception(
            "An error occurred when calculating the minimum dimension.".format(e)
        )

    try:
        # Padding and masking preparation
        t_dict = data_truncator(
            data=t_dict, min_dimension=min_dimension, flag="initial"
        )
        t_dict["min_dimension"] = min_dimension

        # Transform to array
        t_dict = convert_t_dict_key_to_numpy_arrays(dictionary=t_dict, keys=["x", "y"])

        # Normalize the data
        norm_x = DecibelNormalizer(t_dict["x"])
        t_dict["x"], t_dict["min_max_amplitudes"] = (
            norm_x.normalize(),
            norm_x.get_min_max(),
        )
        norm_y = DecibelNormalizer(t_dict["y"])
        t_dict["y"], t_dict["min_max_amplitudes"] = (
            norm_y.normalize(),
            norm_y.get_min_max(),
        )

        # Transform to recarray
        t_dict_recarray = convert_to_recarray(data_dict=t_dict)

        print(f"@@@@@@ Processed files: {data_point_amount} @@@@@@")
        print()

    except Exception as e:
        raise Exception(
            "An error occurred during spectrogram-to-data-format conversion."
        )

    if not save_file_path:
        raise ValueError("An error occurred. 'save_file_path' cannot be empty.")
    # Save normalized data
    savez_numpy_data(file_path=f"{save_file_path}", data=t_dict_recarray)
