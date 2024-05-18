import numpy as np


class DecibelNormalizer:
    def __init__(self, array):
        self.array = array
        self.min_val = None
        self.max_val = None

    def normalize(self):
        # Convert spectrogram magnitudes to decibels
        epsilon = 1e-10  # small constant to avoid log(0)
        self.array = 20 * np.log10(np.maximum(self.array, epsilon))

        # Find min and max values in the dB-scaled array
        self.min_val, self.max_val = np.min(self.array), np.max(self.array)

        # Avoid division-by-zero error in case of all same elements
        if self.min_val == self.max_val:
            self.array[:] = 0
        else:
            self.array = (self.array - self.min_val) / (self.max_val - self.min_val)

        return self.array

    def get_min_max(self):
        return self.min_val, self.max_val

    def denormalize(self, min_val, max_val):
        if min_val is not None and max_val is not None:

            # Apply the inverse of min-max normalization
            self.array = (self.array * (max_val - min_val)) + min_val

            # Convert decibels back to linear scale
            self.array = 10 ** (self.array / 20.0)

        return self.array
