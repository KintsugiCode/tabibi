import librosa
import numpy as np
import matplotlib.pyplot as plt


def audio_to_freq_time_analysis(file_path):
    try:
        # use librosa to load audio file
        signal, sample_rate = librosa.load(file_path, sr=44100, mono=True)

        # STFT -> spectrogram
        hop_length = 80  # in num. of samples
        n_fft = 5000  # window in num. of samples
        n_mels = 320  # number of mel bands to generate

        # perform stft
        stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
        # calculate abs values on complex numbers to get magnitude
        mag_spectrogram, phase = librosa.magphase(stft)

        # Creating filters
        mel_scale = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels)

        # Constructing Mel Spectrogram by matrix multiplying STFT with Mel filters
        mel_spectrogram = np.dot(mel_scale, mag_spectrogram)

        return mel_spectrogram, phase

    except Exception as e:
        raise Exception("Exception occurred: {}".format(e))
