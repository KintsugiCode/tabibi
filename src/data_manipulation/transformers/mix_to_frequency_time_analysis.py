import os
import numpy as np
import librosa
from numpy import savez_compressed
from numpy import asarray

from src.__helpers__.extractor_loader import get_one_file_with_extension
from src.frequency_converter.frequency_time_analysis import audio_to_freq_time_analysis

BASE_PATH = "../../data/raw/V1"
BASE_SAVE_PATH = "../../data/processed/train"

train_dict = {
    "x_train": list(),
    "y_train": list()
}

data_point_amount = 0

for foldername in os.listdir(f"{BASE_PATH}"):
    for mix_file_name in os.listdir(f"{BASE_PATH}/{foldername}/"):
        if mix_file_name.endswith(".wav"):
            print(f"@@ data_point: {mix_file_name}")

            data_point_amount += 1

            mix_folder_path = f"{BASE_PATH}/{foldername}"
            mix_file_path = f"{mix_folder_path}/{mix_file_name}"

            print(mix_file_name)

            bass_folder_path = f"{mix_folder_path}/Bass"
            bass_file_name = get_one_file_with_extension(directory_path=bass_folder_path,
                                                         extension="wav")
            print(bass_file_name)
            print()
            if bass_file_name is None:
                continue
            bass_file_path = f"{bass_folder_path}/{bass_file_name}"

            mix_spectrogram = audio_to_freq_time_analysis(file_path=mix_file_path)
            bass_spectrogram = audio_to_freq_time_analysis(file_path=bass_file_path)
            train_dict["x_train"].append(mix_spectrogram)
            train_dict["y_train"].append(bass_spectrogram)

            # spectogram_nparray = asarray(mix_spectrogram)

            # Create folder structure to store processed data in for one track
            # os.mkdir(f"{BASE_SAVE_PATH}{foldername}/")

            # Save output into file
            # savez_compressed(
            #   f"{BASE_SAVE_PATH}{foldername}/{mix_file_name}", spectogram_nparray
            # )
            if data_point_amount == 10:
                break

print(f"@@@@@@@@@@ Processed wav files: {data_point_amount}")
print(train_dict)
