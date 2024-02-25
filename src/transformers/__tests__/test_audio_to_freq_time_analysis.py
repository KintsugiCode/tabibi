import json
import os
import numpy as np
import pytest
from src.transformers.audio_to_freq_time_analysis import audio_to_freq_time_analysis


dir_path = os.path.dirname(os.path.realpath(__file__))
file_path = os.path.join(
    dir_path,
    "test_data/AClassicEducation_NightOwl/AClassicEducation_NightOwl_MIX.wav",
)

fourierparameters_path = os.path.join(dir_path, "../../config/fourierparameters.json")
with open(fourierparameters_path) as fourierparameters_file:
    fourierparameters = json.load(fourierparameters_file)


def test_valid_file_path():
    """
    Test if the function handles a valid file path correctly.
    """

    try:
        audio_to_freq_time_analysis(file_path)
    except Exception as e:
        pytest.fail(f"Test failed due to {str(e)}")


def test_invalid_file_path():
    """
    Test if the function raises an exception for an invalid file path.
    """
    invalid_file_path = "invalid/audio/file/path.wav"
    with pytest.raises(Exception) as context:
        audio_to_freq_time_analysis(invalid_file_path)
    assert "Exception occurred" in str(context.value)


def test_flag_false_gives_values_below_threshold():
    """
    Test if the function's output contains values below threshold when flag is False.
    """
    output, _ = audio_to_freq_time_analysis(file_path, flag=False)
    assert np.any(output < fourierparameters["audio_amplitude_threshold"])


def test_flag_true_has_values_above_threshold():
    """
    Test if the function's output contains only zeroes and non-zero values that are above the threshold when flag is
    True.
    """
    output, _ = audio_to_freq_time_analysis(file_path, flag=True)
    assert not np.any(
        (0 < output) & (output < fourierparameters["audio_amplitude_threshold"])
    )


def test_audio_to_freq_time_analysis_returns_tuple():
    """
    Test if the function returns a tuple.
    """
    result = audio_to_freq_time_analysis(file_path)
    assert isinstance(result, tuple)


def test_audio_to_freq_time_analysis_output_has_non_negative_values():
    """
    Test if the function's output contains only non-negative values
    """
    output, _ = audio_to_freq_time_analysis(file_path)
    assert (output >= 0).all()


def test_is_output_ndarray():
    """
    Test if the function returns a numpy array
    """
    mel_spectrogram, phase = audio_to_freq_time_analysis(file_path)
    assert isinstance(mel_spectrogram, np.ndarray) and isinstance(phase, np.ndarray)
