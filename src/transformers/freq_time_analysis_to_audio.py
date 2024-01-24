import librosa
import numpy as np


def audio_to_freq_time_analysis(input_file_path, output_file_path):
    try:
        # Use librosa to load audio file
        stft_data = librosa.load(input_file_path)
        # Perform istft
        audio = librosa.istft(stft_data)
        # Save the audio to file
        librosa.output.write_wav(f"{output_file_path}output.wav", audio, sr=22050)

    except Exception as e:
        raise Exception("Exception occurred: {}".format(e))
