import pytest
import os
from unittest.mock import patch
from src.data_manipulation.audio_spectrograms.mixed_signal_to_dict import (
    mixed_signal_to_dict,
)

dir_path = os.path.dirname(os.path.realpath(__file__))
BASE_PATH = os.path.join(
    dir_path,
    "test_data/",
)


@pytest.fixture
def mock_savez_numpy_data():
    with patch(
        "src.data_manipulation.audio_spectrograms.mixed_signal_to_dict.savez_numpy_data"
    ) as mock:
        yield mock


def test_empty_base_path():
    """
    Tests if the function raises an error when base_path is empty.
    """
    with pytest.raises(Exception):
        mixed_signal_to_dict(
            "", ["AClassicEducation_NightOwl", "ElectricBass"], "save/file/path"
        )


def test_file_not_found():
    """
    Tests if the function raises an error when the file to transform is not found.
    """
    with pytest.raises(Exception):
        mixed_signal_to_dict(BASE_PATH, ["file_not_found"], "save/file/path")


def test_no_files_to_transform(mock_savez_numpy_data):
    """
    Tests if the function raises an error when there are no files to transform.
    """
    with pytest.raises(Exception):
        mixed_signal_to_dict(BASE_PATH, [], "save/file/path")


def test_single_file_to_transform(mock_savez_numpy_data):
    """
    Tests if the function works correctly when there is only one file to transform.
    """
    mixed_signal_to_dict(BASE_PATH, ["AClassicEducation_NightOwl"], "save/file/path")
    assert mock_savez_numpy_data.called


def test_multiple_files_to_transform(mock_savez_numpy_data):
    """
    Tests if the function works correctly when there are multiple files to transform.
    """
    mixed_signal_to_dict(
        BASE_PATH,
        ["AClassicEducation_NightOwl", "BigTroubles_Phantom"],
        "save/file/path",
    )
    assert mock_savez_numpy_data.called


def test_invalid_file_format(mock_savez_numpy_data):
    """
    Tests if the function raises an error when the file format is not .wav.
    """
    with pytest.raises(Exception):
        mixed_signal_to_dict(BASE_PATH, ["invalid_format_file"], "save/file/path")


def test_save_file_path_empty():
    """
    Tests if function raises an error when save_file_path is empty.
    """
    with pytest.raises(Exception):
        mixed_signal_to_dict(
            BASE_PATH, ["AClassicEducation_NightOwl", "BigTroubles_Phantom"], ""
        )


def test_save_file_path_invalid():
    """
    Tests if function raises an error when save_file_path is invalid.
    """
    with pytest.raises(Exception):
        mixed_signal_to_dict(
            BASE_PATH,
            ["AClassicEducation_NightOwl", "BigTroubles_Phantom"],
            "path/with\illeg@l/characters",
        )


def test_bass_path_not_found(mock_savez_numpy_data):
    """
    Tests if the function raises an error when the bass wave file is not found.
    """
    with pytest.raises(Exception):
        mixed_signal_to_dict(BASE_PATH, ["file_without_bass"], "save/file/path")


def test_zero_files_to_transform(mock_savez_numpy_data):
    """
    Tests if the function works correctly when files_to_transform list is empty.
    """
    with pytest.raises(Exception):
        mixed_signal_to_dict(BASE_PATH, [], "save/file/path")
