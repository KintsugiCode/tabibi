import json
import os

import numpy as np

from src.transformers.midi_to_piano_roll import midi_to_piano_roll
from src.transformers.piano_roll_to_midi import piano_roll_to_midi


def midi_to_piano_roll_and_back_trial(
    save_folder_path="./outputs/",
):
    piano_roll1 = midi_to_piano_roll("./data/0016_ElectricBass.mid")
    piano_roll2 = midi_to_piano_roll("./data/0062_ElectricBass.mid")

    piano_roll = np.stack((piano_roll1, piano_roll2))

    piano_roll_to_midi(
        piano_roll=piano_roll,
        output_file_path=save_folder_path,
        tag="TRIAL",
        mix_names=["0016.mid", "0062.mid"],
    )


midi_to_piano_roll_and_back_trial()
