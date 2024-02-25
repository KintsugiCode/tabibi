import numpy as np


class Normalizer:
    def __init__(self, array):
        self.array = array
        self.min_val = None
        self.max_val = None

    def normalize(self):
        # Find min and max values of array
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
            self.array = (self.array * (max_val - min_val)) + min_val
        return self.array
