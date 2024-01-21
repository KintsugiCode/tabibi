import librosa
import numpy as np


def audio_to_freq_time_analysis(file_path):
    # use librosa to load audio file

    signal, sample_rate = librosa.load(
        file_path, sr=22050, mono=True
    )

    # STFT -> spectrogram
    hop_length = 512  # in num. of samples
    n_fft = 2048  # window in num. of samples

    # perform stft
    stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)

    # calculate abs values on complex numbers to get magnitude
    spectrogram = np.abs(stft)

    return spectrogram
