import pytest
import numpy as np
from unittest.mock import patch
from src.transformers.freq_time_analysis_to_audio import (
    lowpass_filter,
    butter_lowpass,
    freq_time_analysis_to_audio,
)


def test_lowpass_filter():
    """
    Checks if the lowpass_filter method returns a numpy ndarray.
    """
    data = np.array([1, 2, 3, 4, 5])
    sr = 22050
    cutoff = 1500
    filtered_data = lowpass_filter(data, cutoff, sr)
    assert isinstance(filtered_data, np.ndarray)


def test_butter_lowpass():
    """
    Checks if the butter_lowpass method returns the expected values.
    """
    sr, cutoff, order = 22050, 1500, 5
    b, a = butter_lowpass(cutoff, sr, order)
    assert len(b) == len(a) == order + 1


@pytest.fixture
def setup_data():
    """
    Used to set up data for the test cases.
    """
    mel_spectrogram_array = np.array(
        [[[1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0]]]
    )
    output_file_path = "test_data/audio"
    mix_names = ["audio1.wav", "audio2.wav", "audio3.wav", "audio4.wav", "audio5.wav"]
    min_max_amplitudes = (0, 1)
    tag = "test"
    return (
        mel_spectrogram_array,
        output_file_path,
        mix_names,
        min_max_amplitudes,
        tag,
    )


@patch("librosa.feature.inverse.mel_to_stft")
def test_freq_time_analysis_to_audio(mel_to_stft, setup_data):
    """
    Case checks if the freq_time_analysis_to_audio method raises an exception.
    """
    (
        mel_spectrogram_array,
        output_file_path,
        mix_names,
        min_max_amplitudes,
        tag,
    ) = setup_data
    mel_to_stft.return_value = np.array([1, 2, 3, 4, 5])
    with pytest.raises(Exception):
        freq_time_analysis_to_audio(
            mel_spectrogram_array,
            output_file_path,
            mix_names,
            min_max_amplitudes,
            tag,
        )
