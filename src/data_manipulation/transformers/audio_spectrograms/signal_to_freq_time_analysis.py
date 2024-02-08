import os
import time

from src.__helpers__.__utils__ import (
    convert_t_dict_key_to_numpy_arrays,
    get_one_file_with_extension,
    savez_numpy_data,
    convert_to_recarray,
)
from src.data_manipulation.transformers.normalization.mix_bass_data_normalizer import (
    Normalizer,
)
from src.data_manipulation.transformers.padding.mix_bass_data_padder import data_padder
from src.data_manipulation.transformers.truncating.mix_bass_data_truncator import (
    data_initial_truncator,
)
from src.transformers.audio_to_freq_time_analysis import audio_to_freq_time_analysis


def transform_mix_and_bass_to_spectrogram(
    base_path, files_to_transform, save_file_path
):
    t_dict = {
        "x": list(),
        "y": list(),
        "x_phase": list(),
        "y_phase": list(),
        "mix_name": list(),
        "min_dimension": 0,
    }

    """
    # Uncomment if a pause is needed to prevent computer hardware from becoming overwhelmed
    data_point_multitude = 1
    """

    data_point_amount = 0
    dim = []

    """
    # Iterates over all folders in base_path and checks if the folder is included in the files_to_transform list.
    # If yes, transforms the mix wav file and the bass wav file into spectrograms.
    """
    for foldername in os.listdir(f"{base_path}"):
        if foldername in files_to_transform:
            for mix_file_name in os.listdir(f"{base_path}/{foldername}/"):
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

                    """
                    # Uncomment if a pause is needed to prevent computer hardware from becoming overwhelmed
                    if data_point_amount == (data_point_multitude * 10):
                        print("Waiting for 10 seconds")
                        data_point_multitude += 1
                        time.sleep(10)
                    """

                    bass_file_path = f"{bass_folder_path}/{bass_file_name}"

                    mix_spectrogram, mix_phase = audio_to_freq_time_analysis(
                        file_path=mix_file_path
                    )
                    bass_spectrogram, bass_phase = audio_to_freq_time_analysis(
                        file_path=bass_file_path, flag=True
                    )

                    t_dict["x"].append(mix_spectrogram)
                    t_dict["y"].append(bass_spectrogram)
                    t_dict["x_phase"].append(mix_phase)
                    t_dict["y_phase"].append(bass_phase)
                    t_dict["mix_name"].append(mix_file_name)

                    # Track the dimensions for later padding
                    dim.append(mix_spectrogram.shape[1])

                    # delete variables after use to free up memory
                    del mix_spectrogram
                    del bass_spectrogram
                    del mix_file_name

                    data_point_amount += 1

    try:
        # Save min_dimension to later truncate the dataset again after overall min_dimension of datasets is known
        min_dimension = min(dim)
    except Exception as e:
        raise Exception(
            "The dataset is missing data. E.g. the provided mixes have no accompanying bass tracks.".format(
                e
            )
        )

    # Padding and masking preparation
    t_dict = data_initial_truncator(data=t_dict, min_dimension=min_dimension)
    t_dict["min_dimension"] = min_dimension

    # Transform to recarray
    t_dict = convert_t_dict_key_to_numpy_arrays(dictionary=t_dict, keys=["x", "y"])

    """
    # Save un-normalized data
    savez_numpy_data(file_path=save_file_path, data=t_dict_recarray)
    """

    # Normalize the data
    norm_x = Normalizer(t_dict["x"])
    t_dict["x"], t_dict["min_max_amplitudes"] = norm_x.normalize(), norm_x.get_min_max()
    norm_y = Normalizer(t_dict["y"])
    t_dict["y"], t_dict["min_max_amplitudes"] = norm_y.normalize(), norm_y.get_min_max()
    t_dict_recarray = convert_to_recarray(data_dict=t_dict)

    # Save normalized data
    savez_numpy_data(file_path=f"{save_file_path}", data=t_dict_recarray)

    print(f"@@@@@@@@@@ Processed wav files: {data_point_amount}")
