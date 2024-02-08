# relative paths to dataset as seen from the main.py file
subset = "V2"
BASE_PATH = f"../data/raw/{subset}"

TRAIN_FOLDER_PATH = "../data/processed/train"
TRAIN_FILE_NAME = f"mix_bass_train_data_{subset}TRAIN"

TEST_FOLDER_PATH = "../data/processed/test"
TEST_FILE_NAME = f"mix_bass_test_data_{subset}TEST"


TRAIN_FILE_PATH = f"{TRAIN_FOLDER_PATH}/normalized_{TRAIN_FILE_NAME}.npz"
TEST_FILE_PATH = f"{TEST_FOLDER_PATH}/normalized_{TEST_FILE_NAME}.npz"

TRAINED_AUDIO_FILE_PATH = "../visualization/trained_audio"
PRED_AUDIO_FILE_PATH = "../visualization/predicted_audio"

VISUALIZATION_SAVE_PATH = "../visualization/spectrograms_visualized/outputs"

TRAINED_MODEL_SAVE_PATH = "../models/trained_models/mark1.pt"
