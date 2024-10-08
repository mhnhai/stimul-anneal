from TraversalPath import *
import matplotlib.pyplot as plt


class TSProblem:
    def __init__(self, cities):
        self.cities = cities

    # main implementation of stimulated annealing
    # max_iteration is each iteration in the main loop
    # temperature is the default temperature
    # if stop_iter > 0, the main loop will stop if it didn't find a better solution after stop_iter
    def run(self, max_iteration=100, temperature=1.0, stop_iter=-1, alpha=0.999):
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

            if self._accept_proba(current_result, new_result, temperature) >= random.random():
                current_solution = new_solution
                current_result = new_result
                chain = 0

                if current_result < best_result:
                    best_solution = TraversalPath(current_solution.path[:])
                    best_result = current_result
            else:
                chain += 1

            if chain == stop_iter and stop_iter > 0:
                break

            temperature = self._decrease_temperature(temperature, alpha)

        return best_solution.path, best_result

    @staticmethod
    def _decrease_temperature(temp, alpha) -> float:
        return temp * alpha

    @staticmethod
    def _accept_proba(prev_result, next_res, temperature) -> float:
        # always accept path that's shorter that the previous path
        if next_res < prev_result:
            return 1.0
        try:
            return math.exp((prev_result - next_res)/temperature)
        except ZeroDivisionError:
            return 0.0


def plot_tsp_solution(cities, best_path, best_actual_path):
    plt.figure(figsize=(10, 6))
    plt.grid()
    # Unzip the cities for plotting
    x, y = zip(*cities)

    # Plot the cities
    plt.scatter(x, y, color='blue', s=40, label='Cities')

    # plot the found path
    best_path_full = best_path + [best_path[0]]  # Closing the loop
    path_x, path_y = zip(*best_path_full)
    plt.plot(path_x, path_y, color='orange', linewidth=1.4, label='Best Path Found')
    # plot the best path
    if best_actual_path:
        best_actual_path_full = best_actual_path + [best_actual_path[0]]
        actual_path_x, actual_path_y = zip(*[cities[i-1] for i in best_actual_path_full])
        plt.plot(actual_path_x, actual_path_y, color='green', linewidth=1.4, label='Best Actual Path')

    # Annotate cities
    # for i, city in enumerate(cities):
    #     plt.annotate(f"City {i}", (city[0], city[1]), textcoords="offset points", xytext=(0, 10), ha='center')

    plt.title("Traveling Salesman Problem Solution")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.show()
