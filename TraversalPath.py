import math
import random


class TraversalPath:
    distance_cache = {}

    def __init__(self, path):
        self.path = path

    @staticmethod
    def reverse_subsequence(arr):
        i, j = sorted(random.sample(range(0, len(arr)), 2))
        arr[i:j + 1] = reversed(arr[i:j + 1])
        return arr

    @classmethod
    def get_distance(cls, city_a, city_b) -> float:
        key = tuple(sorted([city_a, city_b]))
        if key not in cls.distance_cache:
            distance = math.sqrt((city_b[0] - city_a[0]) ** 2 + (city_b[1] - city_a[1]) ** 2)
            cls.distance_cache[key] = distance
        return cls.distance_cache[key]

    def mutate(self, temperature, city_num):
        iteration = max(round(temperature * city_num), 1)
        for _ in range(iteration):
            self.path = self.reverse_subsequence(self.path)

    def get_path_length(self):
        total_length = 0
        for i in range(len(self.path) - 1):
            city_a = self.path[i]
            city_b = self.path[i + 1]
            total_length += self.get_distance(city_a, city_b)

        first = self.path[0]
        last = self.path[-1]
        total_length += self.get_distance(first, last)
        return total_length
