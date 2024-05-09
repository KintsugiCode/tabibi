import os
from src.config.constants import (
    PRODUCTION_FOLDER_BASE_PATH,
    PRODUCTION_INPUT_FOLDER_PATH,
    PRODUCTION_OUTPUT_FOLDER_PATH,
)
from src.data_manipulation.transform_data import transform_data
from src.models.audio_separation.separation_manager import separation_manager
from src.models.tab_transcription.transcription_manager import transcription_manager
from src.models.use_models_on_audio import use_models_on_audio
from src.transformers.piano_roll_to_midi import piano_roll_to_midi

dir_path = os.path.dirname(os.path.realpath(__file__))
PATH_TO_AUDIO = os.path.join(dir_path, PRODUCTION_FOLDER_BASE_PATH)


def main():

    print("@@@@@@ SEPARATION TRAINING/TESTING START @@@@@@")
    separation_manager()
    print()
    print("@@@@@@ MODEL SUCCESSFULLY CREATED @@@@@@")


if __name__ == "__main__":
    main()
