import librosa
import numpy as np
import soundfile as sf


INPUT_FILE_PATH = "../../data/raw/V1/MusicDelta_80sRock/MusicDelta_80sRock_MIX.wav"


def fourier_audio_loss(file_path):
    try:
        # use librosa to load audio file
        signal, sample_rate = librosa.load(file_path, sr=22050, mono=True)

        # STFT -> spectrogram
        hop_length = 512  # in num. of samples
        n_fft = 2048  # window in num. of samples
        # perform stft
        stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
        # calculate abs values on complex numbers to get magnitude
        spectrogram = np.abs(stft)

        # inverts stft whilst estimating phase
        audio = librosa.griffinlim(spectrogram)

        # Save the audio to file
        sf.write("./audio_outputs/Track_TEST.wav", audio, 22050)

    except Exception as e:
        raise Exception("Exception occurred: {}".format(e))


fourier_audio_loss(INPUT_FILE_PATH)
