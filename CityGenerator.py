import random
import math


class CityGenerator:
    @staticmethod
    def generate_random_cities(num_cities, min_coord=0, max_coord=100):
        return [(random.uniform(min_coord, max_coord),
                 random.uniform(min_coord, max_coord))
                for _ in range(num_cities)]

    @staticmethod
    def generate_grid_cities(rows, cols, spacing=1):
        return [(x * spacing, y * spacing)
                for y in range(rows)
                for x in range(cols)]

    @staticmethod
    def generate_circular_cities(num_cities, radius=1):
        return [(radius * math.cos(2 * math.pi * i / num_cities),
                 radius * math.sin(2 * math.pi * i / num_cities))
                for i in range(num_cities)]

    @staticmethod
    def generate_clustered_cities(num_clusters, cities_per_cluster, cluster_radius=10, min_coord=0, max_coord=100):
        cities = []
        for _ in range(num_clusters):
            center_x = random.uniform(min_coord, max_coord)
            center_y = random.uniform(min_coord, max_coord)
            for _ in range(cities_per_cluster):
                angle = random.uniform(0, 2 * math.pi)
                r = random.uniform(0, cluster_radius)
                x = center_x + r * math.cos(angle)
                y = center_y + r * math.sin(angle)
                cities.append((x, y))
        return cities