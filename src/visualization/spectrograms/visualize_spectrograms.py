import numpy as np
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
    tag,
):
    """
    Use this function to create graphs that help recognize the training success of the separation model.
    """

    sr = fourierparameters["sample_rate"]
    hop_length = fourierparameters["hop_length"]

    print("@@@@@@ Creating x_train spectrogram @@@@@@")
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        librosa.power_to_db(mel_spectrogram_x_train, ref=np.max),
        y_axis="mel",
        fmax=8000,
        x_axis="time",
        sr=sr,
        hop_length=hop_length,
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Mel spectrogram - x_train - {mix_name[0]}")
    plt.tight_layout()
    plt.savefig(f"{save_folder_path}/{tag}-spectrogram-x_train.png")

    print("@@@@@@ Creating y_train spectrogram @@@@@@")
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        librosa.power_to_db(mel_spectrogram_y_train, ref=np.max),
        y_axis="mel",
        fmax=8000,
        x_axis="time",
        sr=sr,
        hop_length=hop_length,
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Mel spectrogram - {mix_name[0]}")
    plt.tight_layout()
    plt.savefig(f"{save_folder_path}/{tag}-spectrogram-y_train.png")

    print("@@@@@@ Creating y_train_output spectrogram @@@@@@")
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        librosa.power_to_db(mel_spectrogram_y_train_output, ref=np.max),
        y_axis="mel",
        fmax=8000,
        x_axis="time",
        sr=sr,
        hop_length=hop_length,
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Mel spectrogram - {mix_name[0]}")
    plt.tight_layout()
    plt.savefig(f"{save_folder_path}/{tag}-spectrogram-y_train_output.png")
