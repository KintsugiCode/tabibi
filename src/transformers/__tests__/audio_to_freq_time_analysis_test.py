import numpy as np
import pytest

from src.transformers.audio_to_freq_time_analysis import audio_to_freq_time_analysis


def test_audio_to_freq_time_analysis_valid_file():
    # Update the below path to point to an audio .wav file of your choosing.
    # Expected result is based on real spectrogram data
    audio_file_path = (
        "./transformers/__tests__/test_data/AClassicEducation_NightOwl/AClassicEducation_NightOwl_MIX"
        ".wav"
    )
    result = audio_to_freq_time_analysis(audio_file_path)
    assert isinstance(result, np.ndarray)


def test_audio_to_freq_time_analysis_invalid_file():
    # Expect an error when attempting to load a non-audio or non-existent file
    with pytest.raises(Exception):
        audio_to_freq_time_analysis("invalid_file_path")


def test_audio_to_freq_time_analysis_empty_path():
    # Expect an exception when attempting to load an empty path
    with pytest.raises(Exception):
        audio_to_freq_time_analysis("")
