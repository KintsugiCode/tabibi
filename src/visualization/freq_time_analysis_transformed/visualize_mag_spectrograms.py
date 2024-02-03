import numpy as np
from scipy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt
import librosa
import librosa.display

from src.transformers.audio_to_freq_time_analysis import audio_to_freq_time_analysis


def visualize_spectrograms(
    save_folder_path,
    mel_spectrogram_x_train,
    mel_spectrogram_y_train,
    mel_spectrogram_y_train_output,
):
    print("@@@@@@ Creating x_train spectrogram @@@@@@")
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        librosa.power_to_db(mel_spectrogram_x_train, ref=np.max),
        y_axis="mel",
        fmax=8000,
        x_axis="time",
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title("Mel spectrogram - x_train")
    plt.tight_layout()
    # show plots
    plt.savefig(f"{save_folder_path}/spectrogram-x_train.png")
    print("@@@@@@ Completed x_train spectrogram @@@@@@")
    print()
    print("@@@@@@ Creating y_train spectrogram @@@@@@")
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        librosa.power_to_db(mel_spectrogram_y_train, ref=np.max),
        y_axis="mel",
        fmax=8000,
        x_axis="time",
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title("Mel spectrogram - y_train")
    plt.tight_layout()
    # show plots
    plt.savefig(f"{save_folder_path}/spectrogram-y_train.png")
    print("@@@@@@ Completed y_train spectrogram @@@@@@")
    print()
    print("@@@@@@ Creating y_train_output spectrogram @@@@@@")
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        librosa.power_to_db(mel_spectrogram_y_train_output, ref=np.max),
        y_axis="mel",
        fmax=8000,
        x_axis="time",
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title("Mel spectrogram - y_train_output")
    plt.tight_layout()
    # show plots
    plt.savefig(f"{save_folder_path}/spectrogram-y_train_output.png")
    print("@@@@@@ Completed y_train_output spectrogram @@@@@@")
    print()
