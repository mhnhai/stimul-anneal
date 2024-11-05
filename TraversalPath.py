import math
import random
from itertools import repeat
from numba import njit
import numpy as np

@njit
def calculate_path_length(path_indices, num_cities, distance_matrix):
    total_length = 0.0
    for i in range(0, num_cities):
        from_idx = path_indices[i] - 1
        to_idx = path_indices[(i + 1) % num_cities] - 1
        total_length += distance_matrix[from_idx, to_idx]
    return total_length

class TraversalPath:
    distance_matrix = None

    def __init__(self, path):
        self.path = path
        self.num_cities = len(path)

    def mutate(self):
        i, j = sorted(random.sample(range(0, self.num_cities), 2))
        self.path[i:j + 1] = reversed(self.path[i:j + 1])

    @classmethod
    def _get_distance(cls, city_a:set, city_b:set) -> float:
        return cls.distance_matrix[city_a[2]-1, city_b[2]-1]

    @classmethod
    def _pre_calculate_distance_matrix(cls, cities):
        num_cities = len(cities)
        cls.distance_matrix = np.zeros((num_cities, num_cities), dtype=np.float64)

        for i in range(num_cities):
            for j in range(i+1, num_cities):
                city_a, city_b = cities[i], cities[j]
                dist = math.sqrt((city_a[0] - city_b[0])**2 + (city_a[1] - city_b[1])**2)
                cls.distance_matrix[city_a[2]-1, city_b[2]-1] = dist
                cls.distance_matrix[city_b[2]-1, city_a[2]-1] = dist


    def get_path_length(self):
        _x, _y, indices = zip(*self.path)
        ind = np.array(indices, dtype=np.int64)
        return calculate_path_length(ind, len(ind), self.distance_matrix)