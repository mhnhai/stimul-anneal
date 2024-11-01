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

    def generate_branch_and_bound_solution(self) -> TraversalPath:
        num_cities = len(self.cities)
        visited = [False] * num_cities
        min_cost = float('inf')
        final_path = []
        tp = TraversalPath([])

        def calculate_lower_bound(current_cost, current_city, unvisited):
            bound = current_cost
            for city in unvisited:
                min_edge = float('inf')
                for next_city in range(num_cities):
                    if next_city != city and tp.distance_matrix[city][next_city]:
                        min_edge = min(min_edge, tp.distance_matrix[city][next_city])
                if min_edge != float('inf'):
                    bound += min_edge
            return bound

        def backtrack(current_city, count, cost, path):
            nonlocal min_cost, final_path

            if count == num_cities and tp.distance_matrix[current_city][0]:
                total_cost = cost + tp.distance_matrix[current_city][0]
                if total_cost < min_cost:
                    min_cost = total_cost
                    final_path = path + [0]
                return

            unvisited = set(range(num_cities)) - set(path)
            lower_bound = calculate_lower_bound(cost, current_city, unvisited)

            if lower_bound >= min_cost:
                return

            for next_city in range(num_cities):
                if not visited[next_city] and tp.distance_matrix[current_city][next_city]:
                    visited[next_city] = True
                    backtrack(next_city, count + 1,
                              cost + tp.distance_matrix[current_city][next_city],
                              path + [next_city])
                    visited[next_city] = False

        visited[0] = True
        backtrack(0, 1, 0, [0])
        return TraversalPath([self.cities[i] for i in final_path])


    def run(self, max_iteration=100, temperature=1.0, early_return=10000, initial_solution:TraversalPath=None,
            alpha=0.001, beta=0.1) -> TraversalPath:
        current_solution = initial_solution if initial_solution is not None else self.generate_random_solution()
        current_length = current_solution.get_path_length()

        initial_temp = temperature
        chain = 0
        i = 0
        while i < max_iteration:
            new_solution = TraversalPath(current_solution.path[:])
            new_solution.mutate(temperature, self.city_count)
            new_length = new_solution.get_path_length()

            probability = self._accept_proba(current_length, new_length, temperature, beta)
            if 0 < probability < 1:
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
        except:
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
    #ax = fig.add_subplot(2, 2, (1, 3))

    x, y, _z = zip(*cities)

    if best_actual_path:
        _display_actual_path(best_actual_path, cities)

    for path in other_paths:
        _display_other_path(path['cities'], path['name'])

    ax.scatter(x, y, color='slateblue', s=20, label='Cities', zorder=1)
    current_path_line, = ax.plot([], [], color='cornflowerblue', zorder=2, linewidth=1.4, label='Stimulated annealing')
    _display_info()

    ax2 = fig.add_subplot(gs[0, 2:])
    #ax2 = fig.add_subplot(2, 2, 2)
    data = tsp.stored_probability
    ax2.plot(data)
    ax2.set_title('Probability Plot')
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Probability')

    # Distance plot (bottom right)
    ax3 = fig.add_subplot(gs[1, 2:])
    #ax3 = fig.add_subplot(2, 2, 4)
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
