import os
import shutil


path = "../data/raw/model2/combined"


files = os.listdir(path)


# Iterate over each file
for file in files:
    # Check if the file is an audio file
    if file.endswith(".flac"):
        # Extract the leading number from the file name
        number = file.split("_")[0]

        # Create a new directory if it doesn't exist
        if not os.path.exists(os.path.join(path, number)):
            os.makedirs(os.path.join(path, number))

        # Move the audio file
        shutil.move(os.path.join(path, file), os.path.join(path, number, file))

        # Find the corresponding MIDI file
        base_name = file.split("_")[0]
        for midi_file in files:
            if midi_file.startswith(base_name) and midi_file.endswith(".mid"):
                # Move the MIDI file
                shutil.move(
                    os.path.join(path, midi_file), os.path.join(path, number, midi_file)
                )
                break
