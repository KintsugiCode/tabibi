import librosa
import soundfile as sf


def freq_time_analysis_to_audio(spectogram_array, output_file_path):
    try:
        for track in range(spectogram_array.shape[0]):
            # Perform istft directly on each spectogram
            audio = librosa.istft(spectogram_array[track])
            # Save the audio to file
            sf.write(f"{output_file_path}/Track_{track}.wav", audio, 22050)

    except Exception as e:
        raise Exception("Exception occurred: {}".format(e))
