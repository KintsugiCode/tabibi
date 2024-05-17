# ----------
# Production Constants -- REPLACE THESE WITH YOUR OWN PATHS TO YOUR TRAINED MODELS AND YOUR INPUT/OUTPUT FOLDERS
PRODUCTION_FOLDER_BASE_PATH = "./test_audio"
PRODUCTION_INPUT_FOLDER_PATH = "input"
PRODUCTION_OUTPUT_FOLDER_PATH = "output"
SEPARATION_MODEL_PATH = "../models/trained_models/separation/mark1.pt"
TRANSCRIPTION_MODEL_PATH = "../models/trained_models/transcription/mark1.pt"
# ----------

# Model 1 Constants
MODEL1_SUBSET = "V7"
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
TRAINED_MODEL1_SAVE_PATH = "../models/trained_models/separation/mark1.pt"

# Model 2 Constants
MODEL2_SUBSET = "combined"
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
TRAINED_MODEL2_SAVE_PATH = "../models/trained_models/transcription/mark1.pt"

# Visualisation and Testing constants
TRAINED_AUDIO_FILE_PATH = (
    "../../../../Bachelor Thesis/thesis/audio outputs/trained audio"
)
PRED_AUDIO_FILE_PATH = (
    "../../../../Bachelor Thesis/thesis/audio outputs/predicted audio"
)
VISUALIZATION_SAVE_PATH = "../../../../Bachelor Thesis/thesis/spectrograms"
PRED_MIDI_FILE_PATH = "../visualization/midi/predicted_midi"
