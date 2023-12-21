import os
import numpy as np
import librosa
from numpy import savez_compressed
from numpy import asarray


# file location
file = "../src/data/raw/V1/AClassicEducation_NightOwl/AClassicEducation_NightOwl_MIX.wav"

# use librosa to load audio file
signal, sample_rate = librosa.load(file, sr=22050)


# STFT -> spectrogram
hop_length = 512  # in num. of samples
n_fft = 2048  # window in num. of samples

# calculate duration hop length and window in seconds
hop_length_duration = float(hop_length)/sample_rate
n_fft_duration = float(n_fft)/sample_rate


# perform stft
stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)

# calculate abs values on complex numbers to get magnitude
spectrogram = np.abs(stft)

spectogram_nparray = asarray(spectrogram)

print(spectogram_nparray)

# Save output into file
savez_compressed('data.npz', spectogram_nparray)
