from src.__helpers__.constants import (
    MODEL1_BASE_PATH,
    MODEL2_BASE_PATH,
    MODEL1_TRAIN_FILE_PATH,
    MODEL1_TEST_FILE_PATH,
    MODEL2_TRAIN_FILE_PATH,
    MODEL2_TEST_FILE_PATH,
)
from src.data_manipulation.transformers.audio_spectrograms.mixed_signal_to_dict import (
    mixed_signal_to_dict,
)
from src.data_manipulation.transformers.data_splitter.train_test_split import (
    train_test_splitter,
)
from src.data_manipulation.transformers.tab_transcriptions.bass_and_midi_to_dict import (
    bass_and_midi_to_dict,
)


def transform_data(flag):
    if flag == "audio separation":
        BASE_PATH = MODEL1_BASE_PATH
        TRAIN_FILE_PATH = MODEL1_TRAIN_FILE_PATH
        TEST_FILE_PATH = MODEL1_TEST_FILE_PATH

        # Split data into train/test
        print("@@@@@@ Splitting data into train/test @@@@@@")
        train_files, test_files = train_test_splitter(BASE_PATH)

        # Transform training data and receive max_dimension
        print("@@@@@@ Transforming training data @@@@@@")

        # Transform training data
        mixed_signal_to_dict(
            base_path=BASE_PATH,
            files_to_transform=train_files,
            save_file_path=TRAIN_FILE_PATH,
        )
        # Transform testing data
        print("@@@@@@ Transforming testing data @@@@@@")
        mixed_signal_to_dict(
            base_path=BASE_PATH,
            files_to_transform=test_files,
            save_file_path=TEST_FILE_PATH,
        )
    elif flag == "tab transcription":
        BASE_PATH = MODEL2_BASE_PATH
        TRAIN_FILE_PATH = MODEL2_TRAIN_FILE_PATH
        TEST_FILE_PATH = MODEL2_TEST_FILE_PATH

        # Split bass audio into train/test
        print("@@@@@@ Splitting data into train/test @@@@@@")
        train_files, test_files = train_test_splitter(BASE_PATH)

        # Transform training data and receive max_dimension
        print("@@@@@@ Transforming training data @@@@@@")

        # Transform training data
        bass_and_midi_to_dict(
            base_path=BASE_PATH,
            files_to_transform=train_files,
            save_file_path=TRAIN_FILE_PATH,
        )
        # Transform testing data
        print("@@@@@@ Transforming testing data @@@@@@")
        bass_and_midi_to_dict(
            base_path=BASE_PATH,
            files_to_transform=test_files,
            save_file_path=TEST_FILE_PATH,
        )

    else:
        raise Exception("Invalid flag passed to transform_data function.")


transform_data(flag="tab transcription")
