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

    choice = (
        input(
            "@@@@@@ Would you like to create new models before processing your audio? @@@@@@\n"
            '@@@@@@ WARNING: Selecting "yes" requires training/testing datasets to be present in the data/raw/model1 and '
            "data/raw/model2 folders. @@@@@@\n"
            '@@@@@@  Selecting "no" will use the pre-existing mark1 models instead. [Y/N]: @@@@@@ '
        )
        .strip()
        .lower()
    )

    if choice in ["yes", "y"]:
        # Trains and tests both models separately
        print("@@@@@@ SEPARATION TRAINING/TESTING START @@@@@@")
        separation_manager()
        print()
        print("@@@@@@ TRANSCRIPTION TRAINING/TESTING START @@@@@@")
        transcription_manager()
        print()
        print("@@@@@@ MODELS SUCCESSFULLY CREATED @@@@@@")

    elif choice in ["no", "n"]:
        pass

    else:
        raise Exception("Please enter a valid input. Either [Y/N] or [Yes/No].")

    try:
        for track in os.listdir(f"{PATH_TO_AUDIO}/{PRODUCTION_INPUT_FOLDER_PATH}"):
            if track.endswith(".wav"):
                print(f"@@@@ USING MODELS ON SELECTED TRACK: {track} @@@@")
                input_file_path = (
                    f"{PATH_TO_AUDIO}/{PRODUCTION_INPUT_FOLDER_PATH}/{track}"
                )
                output_file_path = (
                    f"{PATH_TO_AUDIO}/{PRODUCTION_OUTPUT_FOLDER_PATH}/{track}"
                )

                print("@@ Pre-Processing... @@")
                # Pre-process the audio into correct spectrogram format
                processed_input = transform_data(
                    flag="production input",
                    file_name=track,
                    input_file_path=input_file_path,
                )

                print("@@ Separating and transcribing... @@")
                # Pass processed audio through both models
                output = use_models_on_audio(processed_input)

                print("@@ Post-Processing... @@")
                # Post-process the midi-spectrogram into pure midi format and save it
                piano_roll_to_midi(
                    piano_roll=output,
                    output_file_path=output_file_path,
                    tag="",
                    mix_names=["mix_name"],
                    flag="production",
                )

            else:
                print(f"@@@@ {track} is not a .wav file. Skipping... @@@@")
    except Exception as e:
        raise Exception(f"No input files present in input folder: {e}")


if __name__ == "__main__":
    main()
