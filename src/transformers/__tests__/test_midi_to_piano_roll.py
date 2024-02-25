import json
import math
import os
import numpy as np
from src.transformers.midi_to_piano_roll import midi_to_piano_roll

dir_path = os.path.dirname(os.path.realpath(__file__))
fourierparameters_path = os.path.join(dir_path, "../../config/fourierparameters.json")
with open(fourierparameters_path) as fourierparameters_file:
    fourierparameters = json.load(fourierparameters_file)


def test_midi_to_piano_roll():
    """
    Tests midi_to_piano_roll function by checking the shape and value range of result
    """
    midi_sample_path = (
        "../transformers/__tests__/test_data/ElectricBass/0016_ElectricBass.mid"
    )

    result = midi_to_piano_roll(midi_sample_path)

    assert result.shape[1] == int(
        math.ceil(
            fourierparameters["track_seconds_considered"]
            / (fourierparameters["hop_length"] / fourierparameters["sample_rate"])
        )
    ), "Incorrect output shape"
    assert np.max(result) <= 1 and np.min(result) >= 0, "Values are not in binary form"
