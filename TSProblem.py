import random
import math
import TraversalPath
import matplotlib.pyplot as plt

class TSProblem:
    def __init__(self, cities):
        self.cities = cities

    # main implementation of stimulated annealing
    # max_iteration is each iteration in the main loop
    # temperature is the default temperature
    # verbose is to print stats to the console if necessary, 0=no print, 1=print
    # if stop_iter > 0, the main loop will stop if it didnt find a better solution after stop_iter
    def run(self, max_iteration=100, temperature=1.0, verbose=0, stop_iter=-1):
        # initialize the path randomly
        current_solution = TraversalPath(random.sample(self.cities, len(self.cities)))
        current_result = current_solution.get_path_length()

        best_solution = current_solution
        best_result = current_result

        chain = 0

        for i in range(max_iteration):
            new_solution = TraversalPath(current_solution.path[:])
            new_solution.mutate()
            new_result = new_solution.get_path_length()

            if self.accept_proba_(current_result, new_result, temperature) >= random.random():
                current_solution = new_solution
                current_result = new_result
                chain = 0
                if verbose > 0:
                    print(f"Iteration {i}\nCurrent solution: {current_solution.path}\nCurrent result: {current_result}")

                if current_result < best_result:
                    best_solution = TraversalPath(current_solution.path[:])
                    best_result = current_result
            else:
                chain += 1

            if chain == stop_iter and stop_iter > 0:
                break

            temperature = self.decrease_temperature_(temperature, i)

        return best_solution.path, best_result

    @staticmethod
    def decrease_temperature_(temp, iteration) -> float:
        return temp / (1 + 0.001 * iteration)

    @staticmethod
    def accept_proba_(prev_result, next_res, temperature) -> float:
        # always accept path that's shorter that the previous path
        if next_res < prev_result:
            return 1.0
        try:
            return math.exp((prev_result - next_res)/temperature)
        except ZeroDivisionError:
            return 0.0

class TraversalPath:
    def __init__(self, path):
        self.path = path

    @staticmethod
    def swap_random_pair(arr):
        i, j = random.sample(range(0, len(arr)), 2)
        arr[i], arr[j] = arr[j], arr[i]
        return arr

    def mutate(self, max_iter=2):
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


def plot_tsp_solution(cities, best_path):
    plt.figure(figsize=(10, 6))
    # Unzip the cities for plotting
    x, y = zip(*cities)

    # Plot the cities
    plt.scatter(x, y, color='blue', s=100, label='Cities')

    # Plot the path
    best_path_full = best_path + [best_path[0]]  # Closing the loop
    path_x, path_y = zip(*best_path_full)
    plt.plot(path_x, path_y, color='orange', linewidth=2, label='Best Path')

    # Annotate cities
    for i, city in enumerate(cities):
        plt.annotate(f"City {i}", (city[0], city[1]), textcoords="offset points", xytext=(0, 10), ha='center')

    plt.title("Traveling Salesman Problem Solution")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid()
    plt.show()