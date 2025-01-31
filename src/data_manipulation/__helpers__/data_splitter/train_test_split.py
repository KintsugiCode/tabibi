from sklearn.model_selection import train_test_split
import os


def train_test_splitter(base_path):
    all_file_names = []
    for foldername in os.listdir(f"{base_path}"):
        for mix_file_name in os.listdir(f"{base_path}/{foldername}/"):
            if mix_file_name.endswith(".wav") or mix_file_name.endswith(".flac"):
                all_file_names.append(foldername)

    # Raise exception if no files found
    if not all_file_names:
        raise FileNotFoundError("No files found in the directory.")

    # Split into train and test
    train_files, test_files = train_test_split(all_file_names, test_size=0.30)

    return train_files, test_files
