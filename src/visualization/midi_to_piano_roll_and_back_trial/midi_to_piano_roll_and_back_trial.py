import numpy as np
from src.transformers.__helpers__.resize_piano_roll import resize_piano_roll
from src.transformers.audio_to_freq_time_analysis import audio_to_freq_time_analysis
from src.transformers.midi_to_piano_roll import midi_to_piano_roll
from src.transformers.piano_roll_to_midi import piano_roll_to_midi


def midi_to_piano_roll_and_back_trial(
    save_folder_path="./outputs/",
):
    """
    Use this function to convert a midi file to a piano roll and back to midi (incl. resizing) as a means to manually
    test/hear/see how this affects the midi.
    """

    piano_roll1 = midi_to_piano_roll("./data/0016_ElectricBass.mid")
    bass_spectrogram1, _ = audio_to_freq_time_analysis(
        file_path="./data/0016_ElectricBass.flac"
    )
    target_length1 = bass_spectrogram1.shape[1]
    resized_piano_roll1 = resize_piano_roll(piano_roll1, target_length1)

    piano_roll2 = midi_to_piano_roll("./data/0062_ElectricBass.mid")
    bass_spectrogram2, _ = audio_to_freq_time_analysis(
        file_path="./data/0062_ElectricBass.flac"
    )
    target_length2 = bass_spectrogram2.shape[1]
    resized_piano_roll2 = resize_piano_roll(piano_roll2, target_length2)

    resized_piano_roll = np.stack((resized_piano_roll1, resized_piano_roll2))

    piano_roll_to_midi(
        piano_roll=resized_piano_roll,
        output_file_path=save_folder_path,
        tag="TRIAL",
        mix_names=["0016.mid", "0062.mid"],
    )
