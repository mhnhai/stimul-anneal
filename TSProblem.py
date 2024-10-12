import math
import random
import numba

from TraversalPath import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class TSProblem:
    def __init__(self, cities):
        self.cities = cities
        # for animation purposes
        self.stored_results = []

    def generate_greedy_solution(self):
        # Start from the first city
        unvisited = self.cities[:]
        current_city = unvisited.pop(0)
        path = [current_city]

        while unvisited:
            # Find the nearest unvisited city
            nearest_city = min(unvisited, key=lambda city: TraversalPath.get_distance(current_city, city))
            path.append(nearest_city)
            unvisited.remove(nearest_city)
            current_city = nearest_city

        return TraversalPath(path)

    def generate_random_solution(self):
        shuffled = self.cities[:]
        random.shuffle(shuffled)
        return TraversalPath(shuffled)

    def run(self, max_iteration=100, temperature=1.0, greedy_solution=False, stop_iter=-1, alpha=0.001, beta=0.1):
        """
        Run the stimulated annealing algorithm
        :param max_iteration: the maximum iteration
        :param temperature: initial temperature
        :param greedy_solution: if True, it uses greedy initial solution instead of random
        :param stop_iter: stops the loop after stop_iter iteration if no better solutions found
        :param alpha: the cooling rate of temperature using a hyperbolic function. higher = slower decay
        :param beta: a scaler of the delta (next - prev distance). lower = lower probability
        :return: (x, y). x is an array of the found path, and y is the path total length
        """
        # initialize the path randomly
        current_solution = self.generate_greedy_solution() if greedy_solution else self.generate_random_solution()
        current_result = current_solution.get_path_length()
        print(f"Initial distance: {current_result}")

        # best_solution = current_solution
        # best_result = current_result

        city_count = len(self.cities)
        initial_temp = temperature

        chain = 0

        for i in range(max_iteration):
            new_solution = TraversalPath(current_solution.path[:])
            new_solution.mutate(temperature, city_count)
            new_result = new_solution.get_path_length()
            if self._accept_proba(current_result, new_result, temperature, beta) >= random.random():
                current_solution = new_solution
                current_result = new_result
                chain = 0

                self.stored_results.append(current_solution.path[:])

                # if current_result < best_result:
                #     best_solution = TraversalPath(current_solution.path[:])
                #     best_result = current_result
            else:
                chain += 1

            if chain == stop_iter and stop_iter > 0:
                break

            temperature = self._decrease_temperature(temperature, alpha, i, initial_temp)

        self.stored_results.append(current_solution.path[:])
        return current_solution.path, current_result

    @staticmethod
    def _decrease_temperature(temp, alpha, iteration, initial_temp) -> float:
        return initial_temp / (1 + (alpha * iteration))

    @staticmethod
    def _accept_proba(prev_res, next_res, temperature, beta) -> float:
        # always accept path that's shorter that the previous path
        if next_res < prev_res:
            return 1.0
        try:
            return 1.0/(1.0 + math.exp((math.sqrt(next_res - prev_res)*beta)/temperature))
        except:
            return 0.0


def plot_tsp_solution(cities, best_path, best_actual_path):
    plt.figure(figsize=(10, 6))
    plt.grid()
    # Unzip the cities for plotting
    x, y = zip(*cities)

    # Plot the cities
    plt.scatter(x, y, color='blue', s=20, alpha=0.7, label='Cities')

    # plot the found path
    best_path_full = best_path + [best_path[0]]  # Closing the loop
    path_x, path_y = zip(*best_path_full)
    plt.plot(path_x, path_y, color='orange', linewidth=1.4, alpha=0.7, label='Best Path Found')
    # plot the best path
    if best_actual_path:
        best_actual_path_full = best_actual_path + [best_actual_path[0]]
        best_actual_path_points = [cities[i] for i in best_actual_path_full]
        actual_path_x, actual_path_y = zip(*best_actual_path_points)
        plt.plot(actual_path_x, actual_path_y, linestyle="-", alpha=0.35, color='green', linewidth=5, label="Best Actual Path")
        pth = TraversalPath(best_actual_path_points)
        print(f"Best actual path distance: {pth.get_path_length()}")

    plt.title("Traveling Salesman Problem Solution")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()

    plt.show()


def plot_tsp_solution_animation(cities, best_actual_path, frames, max_frames=1200):
    if len(frames) > max_frames:
        cropped = random.sample(frames[0:-1], max_frames - 1) + [frames[-1]]
        cropped.sort(key=lambda x: frames.index(x))
        frames = cropped
    print(f"FRAMES: {len(frames)}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.grid()

    x, y = zip(*cities)

    if best_actual_path:
        best_actual_path_full = best_actual_path + [best_actual_path[0]]
        best_actual_path_points = [cities[i] for i in best_actual_path_full]
        actual_path_x, actual_path_y = zip(*best_actual_path_points)
        ax.plot(actual_path_x, actual_path_y, linestyle="-", alpha=0.35, color='lime', linewidth=5, label="Best Actual Path")
        pth = TraversalPath(best_actual_path_points)
        print(f"Best actual path distance: {pth.get_path_length()}")

    ax.scatter(x, y, color='blue', s=20, label='Cities')
    # Create a line object for the current path
    current_path_line, = ax.plot([], [], color='orange', linewidth=1.4, alpha=0.7, label='Current Path')

    plt.title("Traveling Salesman Problem Solution")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()

    def init():
        # Initialize the background of the plot
        current_path_line.set_data([], [])
        return current_path_line,

    def update(frame):
        # Update the line for the current frame
        current_path = frames[frame]
        current_path_full = current_path + [current_path[0]]  # Closing the loop
        path_x, path_y = zip(*current_path_full)
        current_path_line.set_data(path_x, path_y)
        return current_path_line,

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=len(frames), init_func=init, blit=True, repeat=False, interval=1)

    plt.show()