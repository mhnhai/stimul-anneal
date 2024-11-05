from TraversalPath import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import random
from matplotlib.gridspec import GridSpec


class TSProblem:
    def __init__(self, cities):
        self.cities = cities
        self.city_count = len(self.cities)
        self.stored_results = []
        self.stored_probability = []

    def generate_greedy_solution(self) -> TraversalPath:
        unvisited = self.cities[:]
        first_city = unvisited.pop(0)
        current_city = first_city
        path = [current_city]

        while unvisited:
            nearest_city = min(unvisited, key=lambda city: TraversalPath._get_distance(current_city, city))
            path.append(nearest_city)
            unvisited.remove(nearest_city)
            current_city = nearest_city
        return TraversalPath(path)

    def generate_random_solution(self) -> TraversalPath:
        shuffled = self.cities[:]
        random.shuffle(shuffled)
        return TraversalPath(shuffled)

    def run(self, max_iteration=10000, temperature=1.0, early_return=10000,
            initial_solution:TraversalPath=None, alpha=0.001, beta=0.1) -> TraversalPath:
        current_solution = initial_solution if initial_solution is not None else self.generate_random_solution()
        current_length = current_solution.get_path_length()

        initial_temp = temperature
        chain = 0
        i = 0
        while i < max_iteration:
            new_solution = TraversalPath(current_solution.path[:])
            new_solution.mutate()
            new_length = new_solution.get_path_length()

            probability = self._accept_proba(current_length, new_length, temperature, beta)

            if 0.0 < probability < 1.0:
                self.stored_probability.append(probability)

            if probability >= random.random():
                current_solution = new_solution
                current_length = new_length
                self.stored_results.append(current_solution)
                chain = 0
            else:
                chain += 1
                if chain > early_return:
                    break

            temperature = self._decrease_temperature(alpha, i, initial_temp)
            i += 1

        self.stored_results.append(current_solution)
        return current_solution

    @staticmethod
    def _decrease_temperature(alpha, iteration, initial_temp) -> float:
        return initial_temp / (1 + (alpha * iteration))

    @staticmethod
    def _accept_proba(prev_res, next_res, temperature, beta) -> float:
        if next_res < prev_res:
            return 1.0
        try:
            return math.exp(-math.pow(next_res - prev_res, 2) * beta / temperature)
        except ZeroDivisionError:
            return 0.0


def _display_actual_path(best_actual_path, cities):
    path_unclosed = [cities[i] for i in best_actual_path]
    path_closed = path_unclosed + [path_unclosed[0]] # close the loop
    actual_path_x, actual_path_y, _z = zip(*path_closed)
    plt.plot(actual_path_x, actual_path_y, linestyle="-", zorder=0, alpha=0.2,
             linewidth=5, label="Optimal Path")
    pth = TraversalPath(path_unclosed)
    print(f"\nBest optimal path distance: {pth.get_path_length():.4f}")


def _display_other_path(cities, pathname):
    actual_path_x, actual_path_y, _z = zip(*cities)
    plt.plot(actual_path_x, actual_path_y, linestyle="--", alpha=0.3, linewidth=3, label=pathname)


def _display_info():
    plt.title("Traveling Salesman Problem Solution")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid()
    plt.legend()


def plot_tsp_solution_animation(cities, best_actual_path, tsp, max_frames=1200, other_paths=[]):
    frames = tsp.stored_results
    frames = [i.path for i in frames]
    if len(frames) > max_frames:
        cropped = random.sample(frames[0:-1], max_frames - 1) + [frames[-1]]
        cropped.sort(key=lambda x: frames.index(x))
        frames = cropped


    fig = plt.figure(figsize=(18, 9))
    fig.tight_layout()
    gs = GridSpec(2, 3, figure=fig)
    ax = fig.add_subplot(gs[:, 0:2])

    x, y, _z = zip(*cities)

    if best_actual_path:
        _display_actual_path(best_actual_path, cities)

    for path in other_paths:
        _display_other_path(path['cities'], path['name'])

    ax.scatter(x, y, color='slateblue', s=20, label='Cities', zorder=1)
    current_path_line, = ax.plot([], [], color='salmon', zorder=2, linewidth=1.4, label='Stimulated annealing')
    _display_info()

    ax2 = fig.add_subplot(gs[0, 2:])
    data = tsp.stored_probability
    ax2.plot(data)
    ax2.set_title('Probability Plot')
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Probability')

    ax3 = fig.add_subplot(gs[1, 2:])
    res = tsp.stored_results
    data = [i.get_path_length() for i in res]
    ax3.plot(data)
    ax3.set_title('Distance Plot')
    ax3.set_xlabel('Iterations')
    ax3.set_ylabel('Distance')

    def init():
        current_path_line.set_data([], [])
        return current_path_line,

    def update(frame):
        current_path = frames[frame]
        current_path_full = current_path + [current_path[0]]
        path_x, path_y, _z = zip(*current_path_full)
        current_path_line.set_data(path_x, path_y)
        return current_path_line,

    ani = animation.FuncAnimation(fig, update, frames=len(frames), init_func=init, blit=True, repeat=False, interval=1)
    plt.tight_layout()
    plt.get_current_fig_manager().full_screen_toggle()
    plt.show()
