import math
import random

from PyQt6.sip import array
from numba import njit

class TraversalPath:
    distance_matrix = []

    def __init__(self, path):
        self.path = path
        self.num_cities = len(path)

    @staticmethod
    def reverse_subsequence(arr):
        i, j = sorted(random.sample(range(0, len(arr)), 2))
        arr[i:j + 1] = reversed(arr[i:j + 1])
        return arr

    def mutate(self, temperature, city_num):
        iteration = max(round(temperature * city_num), 1)
        for _ in range(iteration):
            self.path = self.reverse_subsequence(self.path)

    @classmethod
    def get_distance(cls, city_a, city_b) -> float:
        return cls.distance_matrix[city_a[2]-1][city_b[2]-1]

    @classmethod
    def _pre_calculate_distance_matrix(cls, cities):
        num_cities = len(cities)
        cls.distance_matrix = [[0] * num_cities for _ in range(num_cities)]

        for i in range(num_cities):
            for j in range(i+1, num_cities):
                city_a, city_b = cities[i], cities[j]
                dist = TraversalPath._quick_distance(city_a[0], city_b[0], city_a[1], city_b[1])
                cls.distance_matrix[city_a[2]-1][city_b[2]-1] = dist
                cls.distance_matrix[city_b[2]-1][city_a[2]-1] = dist

    @staticmethod
    @njit
    def _quick_distance(ax, bx, ay, by):
        return math.sqrt((bx-ax)**2 + (by-ay)**2)

    def get_path_length(self):
        total_length = sum(
            self.get_distance(self.path[i], self.path[i + 1]) for i in range(-1, self.num_cities-1)
        )
        return total_length