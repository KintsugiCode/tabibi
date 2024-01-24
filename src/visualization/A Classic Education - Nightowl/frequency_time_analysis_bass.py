import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

# file location
file = "../src/data/raw/V1/AClassicEducation_NightOwl/AClassicEducation_NightOwl_RAW/AClassicEducation_NightOwl_RAW_01_01.wav"

# use librosa to load audio file
signal, sample_rate = librosa.load(file, sr=22050)


# display waveform
plt.figure()
librosa.display.waveshow(signal, sr=sample_rate, color="blue")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Waveform)")


# STFT -> spectrogram
hop_length = 512  # in num. of samples
n_fft = 2048  # window in num. of samples

# calculate duration hop length and window in seconds
hop_length_duration = float(hop_length) / sample_rate
n_fft_duration = float(n_fft) / sample_rate

print("STFT hop length duration is: {}s".format(hop_length_duration))
print("STFT window duration is: {}s".format(n_fft_duration))

# perform stft
stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)

# calculate abs values on complex numbers to get magnitude
spectrogram = np.abs(stft)

# display spectrogram
plt.figure()
librosa.display.specshow(spectrogram, sr=sample_rate, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("Frequency")

plt.title("Spectrogram")

# apply logarithm to cast amplitude to Decibels
log_spectrogram = librosa.amplitude_to_db(spectrogram)

librosa.display.specshow(log_spectrogram, sr=sample_rate, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar(format="%+2.0f dB")
plt.title("Spectrogram (dB)")


# # MFCCs
# # extract 13 MFCCs
# MFCCs = librosa.feature.mfcc(
#     signal, sample_rate, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)

# # display MFCCs
# librosa.display.specshow(MFCCs, sr=sample_rate, hop_length=hop_length)
# plt.xlabel("Time")
# plt.ylabel("MFCC coefficients")
# plt.colorbar()
# plt.title("MFCCs")

# show plots
plt.show()
