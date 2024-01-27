import numpy as np


class Normalizer:
    def __init__(self, array):
        self.array = array

    def normalize(self):
        # find min and max values of array
        min_val, max_val = np.min(self.array), np.max(self.array)

        # avoid division by zero error in case of all same elements
        if min_val == max_val:
            self.array[:] = 0
        else:
            self.array = (self.array - min_val) / (max_val - min_val)
        return self.array
