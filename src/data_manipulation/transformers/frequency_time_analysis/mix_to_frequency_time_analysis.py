import os
import numpy as np
import librosa
from numpy import savez_compressed
from numpy import asarray


BASE_PATH = "../src/data/raw/V1/"
BASE_SAVE_PATH = "../src/data/processed/train/"

fileAmount = 0

for foldername in os.listdir(f"{BASE_PATH}"):
    try:
        for filename in os.listdir(f"{BASE_PATH}/{foldername}/"):
            if filename.endswith(".wav"):
                print(f"@@ FILENAME: {filename}")

                fileAmount += 1

                # use librosa to load audio file
                signal, sample_rate = librosa.load(
                    f"{BASE_PATH}/{foldername}/{filename}", sr=22050
                )

                # STFT -> spectrogram
                hop_length = 512  # in num. of samples
                n_fft = 2048  # window in num. of samples

                # calculate duration hop length and window in seconds
                hop_length_duration = float(hop_length) / sample_rate
                n_fft_duration = float(n_fft) / sample_rate

                # perform stft
                stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)

                # calculate abs values on complex numbers to get magnitude
                spectrogram = np.abs(stft)

                spectogram_nparray = asarray(spectrogram)

                print(spectogram_nparray)

                # Create folder structure to store processed data in for one track
                os.mkdir(f"{BASE_SAVE_PATH}{foldername}/")

                # Save output into file
                savez_compressed(
                    f"{BASE_SAVE_PATH}{foldername}/{filename}", spectogram_nparray
                )

                continue
            else:
                continue
    except Exception:
        continue

print(f"@@@@@@@@@@ Processed wav files: {fileAmount}")
