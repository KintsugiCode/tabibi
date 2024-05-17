import os


path = "../data/raw/model1/V2"


folders = os.listdir(path)

counter = 1
# Iterate over each folder
for folder in folders:
    folder_path = os.path.join(path, folder)
    # Rename mixture.wav file
    mixture_file_path = os.path.join(folder_path, "mixture.wav")
    if os.path.exists(mixture_file_path):
        new_mixture_file_name = f"Track{counter}.wav"
        new_mixture_file_path = os.path.join(folder_path, new_mixture_file_name)

        # Rename the mixture.wav file
        os.rename(mixture_file_path, new_mixture_file_path)
        print(f"Renamed {mixture_file_path} to {new_mixture_file_path}")

    # Rename Bass/Bass.wav file
    bass_folder_path = os.path.join(folder_path, "Bass")
    bass_file_path = os.path.join(bass_folder_path, "bass.wav")
    if os.path.exists(bass_file_path):
        new_bass_file_name = f"Bass{counter}.wav"
        new_bass_file_path = os.path.join(bass_folder_path, new_bass_file_name)

        # Rename the Bass.wav file
        os.rename(bass_file_path, new_bass_file_path)
        print(f"Renamed {bass_file_path} to {new_bass_file_path}")

    # Increment the counter
    counter += 1
