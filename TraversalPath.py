import math
import random


class TraversalPath:
    # static
    distance_cache = {}

    def __init__(self, path):
        self.path = path

    @staticmethod
    def swap_random_pair(arr):
        i, j = random.sample(range(0, len(arr)), 2)
        arr[i], arr[j] = arr[j], arr[i]
        return arr

    @staticmethod
    def reverse_subsequence(arr):
        """
        Randomize the array by choosing a random subsequence and reverse it.
        This is 2-opt switching in TSP.
        """
        i, j = sorted(random.sample(range(0, len(arr)), 2))
        arr[i:j + 1] = reversed(arr[i:j + 1])
        return arr

    @classmethod
    def get_distance(cls, city_a, city_b) -> float:
        """
        Calculate the distance of two points. Results are cached
        """
        key = tuple(sorted([city_a, city_b]))
        if key not in cls.distance_cache:
            distance = math.sqrt((city_b[0] - city_a[0]) ** 2 + (city_b[1] - city_a[1]) ** 2)
            cls.distance_cache[key] = distance
        return cls.distance_cache[key]

    def mutate(self, temperature, city_num):
        """
        Randomize the solution. Higher temperature means more randomness.
        :param temperature: the current temperature
        :param city_num: the total city of the problem
        """
        iteration = max(round(temperature * city_num), 1)
        for _ in range(iteration):
            self.path = self.reverse_subsequence(self.path)

    def get_path_length(self):
        """
        Returns path length of this solution
        """
        total_length = 0
        for i in range(len(self.path) - 1):
            city_a = self.path[i]
            city_b = self.path[i + 1]
            total_length += self.get_distance(city_a, city_b)

        first = self.path[0]
        last = self.path[-1]
        total_length += self.get_distance(first, last)
        return total_length
