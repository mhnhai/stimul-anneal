from TraversalPath import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import random
from numba import njit


class TSProblem:
    def __init__(self, cities):
        self.cities = cities
        self.stored_results = []
        self.stored_probability = []

    def generate_greedy_solution(self):
        unvisited = self.cities[:]
        current_city = unvisited.pop(0)
        path = [current_city]

        while unvisited:
            nearest_city = min(unvisited, key=lambda city: TraversalPath.get_distance(current_city, city))
            path.append(nearest_city)
            unvisited.remove(nearest_city)
            current_city = nearest_city

        return TraversalPath(path)

    def generate_random_solution(self):
        shuffled = self.cities[:]
        random.shuffle(shuffled)
        return TraversalPath(shuffled)

    def run(self, max_iteration=100, temperature=1.0, greedy_solution=False, alpha=0.001, beta=0.1):
        current_solution = self.generate_greedy_solution() if greedy_solution else self.generate_random_solution()
        current_result = current_solution.get_path_length()
        print(f"Initial distance: {current_result}")

        city_count = len(self.cities)
        initial_temp = temperature

        for i in range(max_iteration):
            new_solution = TraversalPath(current_solution.path[:])
            new_solution.mutate(temperature, city_count)
            new_result = new_solution.get_path_length()

            probability = self._accept_proba(current_result, new_result, temperature, beta)
            if probability > 0 and probability < 1:
                self.stored_probability.append(probability)

            if probability >= random.random():
                current_solution, current_result = new_solution, new_result
                self.stored_results.append(current_solution)

            temperature = self._decrease_temperature(temperature, alpha, i, initial_temp)

        self.stored_results.append(current_solution)
        return current_solution.path, current_result

    @staticmethod
    @njit
    def _decrease_temperature(temp, alpha, iteration, initial_temp) -> float:
        return initial_temp / (1 + (alpha * iteration))

    @staticmethod
    @njit
    def _accept_proba(prev_res, next_res, temperature, beta) -> float:
        if next_res < prev_res:
            return 1.0
        try:
            #print((next_res - prev_res) * beta)
            result = (math.exp(-math.pow(next_res - prev_res, 2) * beta / temperature))
            return result
        except:
            return 0.0


def _display_actual_path(best_actual_path, cities):
    best_actual_path_full = best_actual_path + [best_actual_path[0]]
    best_actual_path_points = [cities[i] for i in best_actual_path_full]
    actual_path_x, actual_path_y = zip(*best_actual_path_points)
    plt.plot(actual_path_x, actual_path_y, linestyle="-", alpha=0.2, color='green',
             linewidth=5, label="Best Actual Path")
    pth = TraversalPath(best_actual_path_points)
    print(f"Best actual path distance: {pth.get_path_length()}")


def _display_info():
    plt.title("Traveling Salesman Problem Solution")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid()
    plt.legend()


def plot_tsp_solution(cities, best_path, best_actual_path):
    plt.figure(figsize=(9, 9))
    x, y = zip(*cities)

    plt.scatter(x, y, color='blue', s=20, alpha=0.7, label='Cities')

    best_path_full = best_path + [best_path[0]]
    path_x, path_y = zip(*best_path_full)
    plt.plot(path_x, path_y, color='orange', linewidth=1.4, alpha=0.7, label='Best Path Found')

    if best_actual_path:
        _display_actual_path(best_actual_path, cities)

    _display_info()
    plt.tight_layout()
    plt.show()


def plot_tsp_solution_animation(cities, best_actual_path, frames, max_frames=1200):
    frames = [i.path for i in frames]
    if len(frames) > max_frames:
        cropped = random.sample(frames[0:-1], max_frames - 1) + [frames[-1]]
        cropped.sort(key=lambda x: frames.index(x))
        frames = cropped
    print(f"FRAMES: {len(frames)}")

    fig, ax = plt.subplots(figsize=(9, 9))

    x, y = zip(*cities)

    if best_actual_path:
        _display_actual_path(best_actual_path, cities)

    ax.scatter(x, y, color='blue', s=20, label='Cities')
    current_path_line, = ax.plot([], [], color='orange', linewidth=1.4, label='Current Path')
    _display_info()

    def init():
        current_path_line.set_data([], [])
        return current_path_line,

    def update(frame):
        current_path = frames[frame]
        current_path_full = current_path + [current_path[0]]
        path_x, path_y = zip(*current_path_full)
        current_path_line.set_data(path_x, path_y)
        return current_path_line,

    ani = animation.FuncAnimation(fig, update, frames=len(frames), init_func=init, blit=True, repeat=False, interval=1)
    plt.tight_layout()
    plt.show()


def create_proba_plot(tsp: TSProblem):
    plt.figure()  # Create a new figure for the probability plot
    data = tsp.stored_probability
    plt.tight_layout()
    plt.plot(data)
    plt.title('Probability Plot')  # Optional: Add a title
    plt.xlabel('X-axis label')      # Optional: Add x-axis label
    plt.ylabel('Probability')        # Optional: Add y-axis label                    # Display the plot

def create_distance_plot(tsp: TSProblem):
    plt.figure()  # Create a new figure for the distance plot
    res = tsp.stored_results
    data = [i.get_path_length() for i in res]
    plt.tight_layout()
    plt.plot(data)
    plt.title('Distance Plot')       # Optional: Add a title
    plt.xlabel('X-axis label')        # Optional: Add x-axis label
    plt.ylabel('Distance')            # Optional: Add y-axis label