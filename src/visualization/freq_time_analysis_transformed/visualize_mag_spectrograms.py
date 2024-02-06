import numpy as np
from scipy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
import json


dir_path = os.path.dirname(os.path.realpath(__file__))

fourierparameters_path = os.path.join(dir_path, "../../config/fourierparameters.json")

with open(fourierparameters_path) as fourierparameters_file:
    fourierparameters = json.load(fourierparameters_file)


def visualize_spectrograms(
    save_folder_path,
    mel_spectrogram_x_train,
    mel_spectrogram_y_train,
    mel_spectrogram_y_train_output,
    mix_name,
):
    frame_times = librosa.frames_to_time(np.arange(mel_spectrogram_x_train.shape[1]))
    sr = fourierparameters["sample_rate"]
    hop_length = fourierparameters["hop_length"]

    print("@@@@@@ Creating x_train spectrogram @@@@@@")
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        librosa.power_to_db(mel_spectrogram_x_train, ref=np.max),
        x_coords=frame_times,
        y_axis="mel",
        fmax=8000,
        x_axis="time",
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Mel spectrogram - x_train - {mix_name[0]}")
    plt.tight_layout()
    # show plots
    plt.savefig(f"{save_folder_path}/spectrogram-x_train.png")
    print("@@@@@@ Completed x_train spectrogram @@@@@@")
    print("@@@@@@ Creating y_train spectrogram @@@@@@")
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        librosa.power_to_db(mel_spectrogram_y_train, ref=np.max),
        x_coords=frame_times,
        y_axis="mel",
        fmax=8000,
        x_axis="time",
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Mel spectrogram - {mix_name[0]}")
    plt.tight_layout()
    # show plots
    plt.savefig(f"{save_folder_path}/spectrogram-y_train.png")
    print("@@@@@@ Completed y_train spectrogram @@@@@@")
    print("@@@@@@ Creating y_train_output spectrogram @@@@@@")
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        librosa.power_to_db(mel_spectrogram_y_train_output, ref=np.max),
        x_coords=frame_times,
        y_axis="mel",
        fmax=8000,
        x_axis="time",
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Mel spectrogram - {mix_name[0]}")
    plt.tight_layout()
    # show plots
    plt.savefig(f"{save_folder_path}/spectrogram-y_train_output.png")
    print("@@@@@@ Completed y_train_output spectrogram @@@@@@")
