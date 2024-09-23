import matplotlib.pyplot as plt
from TSProblem import TSProblem, plot_tsp_solution
from CityGenerator import *

city_gen = CityGenerator()
cities = city_gen.generate_random_cities(40)
tsp = TSProblem(cities)
best_path, best_length = tsp.run(max_iteration=10000, temperature=1.0)
print("Best path:", best_path)
print("Best path length:", best_length)

plot_tsp_solution(cities=cities, best_path=best_path)
