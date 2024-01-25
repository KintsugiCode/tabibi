from sklearn.model_selection import train_test_split
import os


def train_test_splitter(base_path):
    all_file_names = []
    for foldername in os.listdir(f"{base_path}"):
        for mix_file_name in os.listdir(f"{base_path}/{foldername}/"):
            if mix_file_name.endswith(".wav"):
                all_file_names.append(foldername)

    # Split into train and test
    train_files, test_files = train_test_split(all_file_names, test_size=0.30)

    return train_files, test_files
