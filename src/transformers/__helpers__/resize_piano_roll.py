import numpy as np


def resize_piano_roll(piano_roll, target_length):
    num_notes, _ = piano_roll.shape
    resized_piano_roll = np.zeros((num_notes, target_length))

    # Resample the piano roll to fit the target length
    for i in range(num_notes):
        resized_piano_roll[i, :] = np.mean(
            piano_roll[i].reshape(-1, len(piano_roll[i]) // target_length), axis=1
        )

    return resized_piano_roll
