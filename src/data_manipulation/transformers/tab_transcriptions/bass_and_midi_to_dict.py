import os

from src.__helpers__.__utils__ import (
    get_one_file_with_extension,
    convert_t_dict_key_to_numpy_arrays,
    convert_to_recarray,
    savez_numpy_data,
)
from src.data_manipulation.transformers.normalization.mix_bass_data_normalizer import (
    Normalizer,
)
from src.data_manipulation.transformers.truncating.mix_bass_data_truncator import (
    data_initial_truncator,
)
from src.transformers.__helpers__.resize_piano_roll import resize_piano_roll
from src.transformers.audio_to_freq_time_analysis import audio_to_freq_time_analysis
from src.transformers.midi_to_piano_roll import midi_to_piano_roll


def bass_and_midi_to_dict(base_path, files_to_transform, save_file_path):
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
    for foldername in os.listdir(f"{base_path}"):
        if foldername in files_to_transform:
            for file_name in os.listdir(f"{base_path}/{foldername}/"):
                print(f"@@ data_point: {file_name} @@ ")
                file_path = f"{base_path}/{foldername}/{file_name}"
                if file_name.endswith(".flac"):
                    bass_spectrogram, _ = audio_to_freq_time_analysis(
                        file_path=file_path
                    )

                elif file_name.endswith(".mid"):
                    midi_piano_roll = midi_to_piano_roll(file_path=file_path)

                else:
                    raise Exception("Unknown file extension type")

            target_length = bass_spectrogram.shape[1]
            resized_piano_roll = resize_piano_roll(midi_piano_roll, target_length)

            t_dict["x"].append(bass_spectrogram)
            t_dict["y"].append(resized_piano_roll)
            t_dict["mix_name"].append(file_name)

            # Track the dimensions for later padding
            dim.append(bass_spectrogram.shape[1])

            # delete variables after use to free up memory
            del bass_spectrogram
            del midi_piano_roll
            del resized_piano_roll
            del file_name

            """
              # Uncomment if a pause is needed to prevent computer hardware from becoming overwhelmed
              if data_point_amount == (data_point_multitude * 10):
                  print("Waiting for 10 seconds")
                  data_point_multitude += 1
                  time.sleep(10)
            """
            data_point_amount += 1
        if data_point_amount == 5:
            break

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

    # Normalize the bass spectrograms
    norm_x = Normalizer(t_dict["x"])
    t_dict["x"], t_dict["min_max_amplitudes"] = norm_x.normalize(), norm_x.get_min_max()
    t_dict_recarray = convert_to_recarray(data_dict=t_dict)

    # Save normalized data
    savez_numpy_data(file_path=f"{save_file_path}", data=t_dict_recarray)

    print(f"@@@@@@@@@@ Processed files: {data_point_amount}")
