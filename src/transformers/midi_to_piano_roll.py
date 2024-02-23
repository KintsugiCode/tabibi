import json
import os

import pretty_midi
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))

fourierparameters_path = os.path.join(dir_path, "../config/fourierparameters.json")

with open(fourierparameters_path) as fourierparameters_file:
    fourierparameters = json.load(fourierparameters_file)

time_resolution = fourierparameters["hop_length"] / fourierparameters["sample_rate"]


def midi_to_piano_roll(file_path, fs=fourierparameters["sample_rate"]):
    midi_data = pretty_midi.PrettyMIDI(file_path)

    # Calculate start and end time based on audio processing parameters
    start_time = (
        midi_data.get_end_time() - fourierparameters["track_seconds_considered"]
    ) / 2
    end_time = start_time + fourierparameters["track_seconds_considered"]

    times = np.arange(start_time, end_time, time_resolution)

    # Extract piano roll data for the specified time segment
    piano_roll = midi_data.get_piano_roll(fs=fs, times=times)

    # Convert velocity values to binary on/off
    piano_roll = np.where(piano_roll > 0, 1, 0)

    return piano_roll
