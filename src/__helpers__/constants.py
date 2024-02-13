# Model 1 Constants - relative paths to datasets as seen from the main.py file
MODEL1_SUBSET = "V2"
MODEL1_BASE_PATH = f"../data/raw/model1/{MODEL1_SUBSET}"

MODEL1_TRAIN_FOLDER_PATH = "../data/processed/train/model1"
MODEL1_TRAIN_FILE_NAME = f"mix_bass_train_data_{MODEL1_SUBSET}-TRAIN"

MODEL1_TEST_FOLDER_PATH = "../data/processed/test/model1"
MODEL1_TEST_FILE_NAME = f"mix_bass_test_data_{MODEL1_SUBSET}-TEST"


MODEL1_TRAIN_FILE_PATH = (
    f"{MODEL1_TRAIN_FOLDER_PATH}/normalized_{MODEL1_TRAIN_FILE_NAME}.npz"
)
MODEL1_TEST_FILE_PATH = (
    f"{MODEL1_TEST_FOLDER_PATH}/normalized_{MODEL1_TEST_FILE_NAME}.npz"
)


# Model 2 Constants - relative paths to datasets as seen from the main.py file
MODEL2_SUBSET = ""
MODEL2_BASE_PATH = f"../data/raw/model2/{MODEL2_SUBSET}"

MODEL2_TRAIN_FOLDER_PATH = "../data/processed/train/model2"
MODEL2_TRAIN_FILE_NAME = f"mix_bass_train_data_{MODEL2_SUBSET}-TRAIN"

MODEL2_TEST_FOLDER_PATH = "../data/processed/test/model2"
MODEL2_TEST_FILE_NAME = f"mix_bass_test_data_{MODEL2_SUBSET}-TEST"


MODEL2_TRAIN_FILE_PATH = (
    f"{MODEL2_TRAIN_FOLDER_PATH}/normalized_{MODEL2_TRAIN_FILE_NAME}.npz"
)
MODEL2_TEST_FILE_PATH = (
    f"{MODEL2_TEST_FOLDER_PATH}/normalized_{MODEL2_TEST_FILE_NAME}.npz"
)


# Visualisation and Testing constants
TRAINED_AUDIO_FILE_PATH = "../visualization/trained_audio"
PRED_AUDIO_FILE_PATH = "../visualization/predicted_audio"
VISUALIZATION_SAVE_PATH = "../visualization/spectrograms_visualized/outputs"
TRAINED_MODEL_SAVE_PATH = "../models/trained_models/mark1.pt"
