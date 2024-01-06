import librosa
import numpy as np
from scipy.io import wavfile
from scipy import signal


def audio_to_freq_time_analysis(file_path):
    # use librosa to load audio file

    sample_rate, signal = wavfile.read(
        file_path
    )

    # STFT -> spectrogram
    hop_length = 512  # in num. of samples
    n_fft = 2048  # window in num. of samples

    # perform stft
    stft = signal.stft(x=signal, window=n_fft, hop_length=hop_length)

    # calculate abs values on complex numbers to get magnitude
    spectrogram = np.abs(stft)

    return spectrogram





