import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
import json

from src.transformers.__helpers__.resize_piano_roll import resize_piano_roll
from src.transformers.audio_to_freq_time_analysis import audio_to_freq_time_analysis
from src.transformers.midi_to_piano_roll import midi_to_piano_roll

dir_path = os.path.dirname(os.path.realpath(__file__))

fourierparameters_path = os.path.join(dir_path, "../../config/fourierparameters.json")

with open(fourierparameters_path) as fourierparameters_file:
    fourierparameters = json.load(fourierparameters_file)

BASE_PATH = "../../data/raw/model2/combined"
FOLDERNAME = "0095"


def freq_time_analysis(save_folder_path="./TRIAL-midi-to-spectrogram"):
    t_dict = {
        "x": list(),
        "y": list(),
        "mix_name": list(),
        "min_dimension": 0,
    }

    sr = fourierparameters["sample_rate"]
    hop_length = fourierparameters["hop_length"]

    bass_spectrogram, midi_piano_roll = None, None
    for file_name in sorted(os.listdir(f"{BASE_PATH}/{FOLDERNAME}/")):
        print(f"@@ data_point: {file_name} @@ ")
        file_path = f"{BASE_PATH}/{FOLDERNAME}/{file_name}"
        try:
            if file_name.endswith(".flac"):
                bass_spectrogram, _ = audio_to_freq_time_analysis(file_path=file_path)

            elif file_name.endswith(".mid"):
                midi_piano_roll = midi_to_piano_roll(file_path=file_path)

        except Exception as e:
            print("An error occurred in processing {} : {}".format(file_name, e))

        if bass_spectrogram is not None and midi_piano_roll is not None:
            target_length = bass_spectrogram.shape[1]
            resized_piano_roll = resize_piano_roll(midi_piano_roll, target_length)

            t_dict["x"].append(bass_spectrogram)
            t_dict["y"].append(resized_piano_roll)
            t_dict["mix_name"].append(file_name)

            print("@@@@@@ Creating bass spectrogram @@@@@@")
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(
                librosa.power_to_db(bass_spectrogram, ref=np.max),
                y_axis="mel",
                fmax=8000,
                x_axis="time",
                sr=sr,
                hop_length=hop_length,
            )
            plt.colorbar(format="%+2.0f dB")
            plt.title(f"Mel spectrogram - {file_name}")
            plt.tight_layout()
            # save plots
            plt.savefig(f"{save_folder_path}/BASS-{file_name}-spectrogram.png")
            print("@@@@@@ Completed bass spectrogram @@@@@@")

            print("@@@@@@ Creating midi spectrogram @@@@@@")
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(
                librosa.power_to_db(resized_piano_roll, ref=np.max),
                y_axis="mel",
                fmax=8000,
                x_axis="time",
                sr=sr,
                hop_length=hop_length,
            )
            plt.colorbar(format="%+2.0f dB")
            plt.title(f"Mel spectrogram - {file_name}")
            plt.tight_layout()
            # save plots
            plt.savefig(f"{save_folder_path}/MIDI-{file_name}-spectrogram.png")
            print("@@@@@@ Completed midi spectrogram @@@@@@")
        else:
            print("Missing data for: {}".format(file_name))


freq_time_analysis(save_folder_path="./outputs")
