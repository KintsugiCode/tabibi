from src.config.constants import (
    MODEL1_BASE_PATH,
    MODEL2_BASE_PATH,
    MODEL1_TRAIN_FILE_PATH,
    MODEL1_TEST_FILE_PATH,
    MODEL2_TRAIN_FILE_PATH,
    MODEL2_TEST_FILE_PATH,
)
from src.data_manipulation.for_production.production_audio_to_dict import (
    production_audio_to_dict,
)
from src.data_manipulation.for_training.audio_spectrograms.mixed_signal_to_dict import (
    mixed_signal_to_dict,
)
from src.data_manipulation.__helpers__.data_splitter.train_test_split import (
    train_test_splitter,
)
from src.data_manipulation.for_training.tab_transcriptions.bass_and_midi_to_dict import (
    bass_and_midi_to_dict,
)


def transform_data(flag, input_file_path="", file_name=""):
    if flag == "audio separation":
        # Split data into train/test
        print("@@@@ Splitting data into train/test @@@@")
        train_files, test_files = train_test_splitter(MODEL1_BASE_PATH)

        # Transform training data and receive max_dimension
        print("@@@@ Transforming training data @@@@")

        # Transform training data
        mixed_signal_to_dict(
            base_path=MODEL1_BASE_PATH,
            files_to_transform=train_files,
            save_file_path=MODEL1_TRAIN_FILE_PATH,
        )
        # Transform testing data
        print("@@@@ Transforming testing data @@@@")
        mixed_signal_to_dict(
            base_path=MODEL1_BASE_PATH,
            files_to_transform=test_files,
            save_file_path=MODEL1_TEST_FILE_PATH,
        )
    elif flag == "tab transcription":
        # Split bass audio into train/test
        print("@@@@ Splitting data into train/test @@@@")
        train_files, test_files = train_test_splitter(MODEL2_BASE_PATH)

        # Transform training data and receive max_dimension
        print("@@@@ Transforming training data @@@@")

        # Transform training data
        bass_and_midi_to_dict(
            base_path=MODEL2_BASE_PATH,
            files_to_transform=train_files,
            save_file_path=MODEL2_TRAIN_FILE_PATH,
        )
        # Transform testing data
        print("@@@@ Transforming testing data @@@@")
        bass_and_midi_to_dict(
            base_path=MODEL2_BASE_PATH,
            files_to_transform=test_files,
            save_file_path=MODEL2_TEST_FILE_PATH,
        )
    elif flag == "production input":
        # Transform production data
        processed_input = production_audio_to_dict(
            file_name=file_name,
            input_path=input_file_path,
        )
        return processed_input
    else:
        raise Exception("Invalid flag passed to transform_data function.")
