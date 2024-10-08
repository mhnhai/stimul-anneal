import math
import random


class TraversalPath:
    def __init__(self, path):
        self.path = path

    @staticmethod
    def swap_random_pair(arr):
        i, j = random.sample(range(0, len(arr)), 2)
        arr[i], arr[j] = arr[j], arr[i]
        return arr

    def mutate(self, max_iter=1):
        for i in range(max_iter):
            self.path = self.swap_random_pair(self.path)

    def get_path_length(self):
        total_length = 0
        for i in range(len(self.path) - 1):
            city_a = self.path[i]
            city_b = self.path[i + 1]
            total_length += math.sqrt((city_b[0] - city_a[0]) ** 2 + (city_b[1] - city_a[1]) ** 2)

        first = self.path[0]
        last = self.path[-1]
        total_length += math.sqrt((first[0] - last[0]) ** 2 + (first[1] - last[1]) ** 2)
        return total_length
