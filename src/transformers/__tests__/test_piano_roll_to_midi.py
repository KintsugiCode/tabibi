from unittest.mock import patch

import numpy as np
import pytest

from src.transformers.piano_roll_to_midi import piano_roll_to_midi


def test_piano_roll_to_midi_raises_exception():
    """
    Tests piano_roll_to_midi function to ensure it raises an exception when incorrect data is provided
    """
    piano_roll = np.zeros((3, 3, 4))
    output_file_path = ""
    tag = "test_tag"
    mix_names = ["mix_1", "mix_2", "mix_3", "mix4"]

    with pytest.raises(Exception):
        piano_roll_to_midi(piano_roll, output_file_path, tag, mix_names)


@patch("pretty_midi.PrettyMIDI", autospec=True)
def test_piano_roll_to_midi_called_once(mock_pretty_midi):
    """
    Tests piano_roll_to_midi function to ensure PrettyMIDI constructor is called only once for a track
    """
    piano_roll = np.zeros((1, 5, 5))
    output_file_path = "../temp/"
    tag = "test_tag"
    mix_names = ["mix"]

    mock_pretty_midi.return_value.instruments = []

    piano_roll_to_midi(piano_roll, output_file_path, tag, mix_names)
    assert mock_pretty_midi.call_count == 1


@patch("pretty_midi.PrettyMIDI", autospec=True)
def test_piano_roll_to_midi_notes(mock_pretty_midi):
    """
    Tests piano_roll_to_midi function to ensure notes are correctly added to the instrument when a pitch is played
    """
    piano_roll = np.zeros((1, 5, 5))
    piano_roll[0, 2, 0] = 1
    piano_roll[0, 2, 2] = 1
    output_file_path = "../temp/"
    tag = "test_tag"
    mix_names = ["mix"]

    mock_pretty_midi.return_value.instruments = []

    piano_roll_to_midi(piano_roll, output_file_path, tag, mix_names)
    assert len(mock_pretty_midi.return_value.instruments[0].notes) == 2
