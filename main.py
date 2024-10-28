from TSProblem import TSProblem, plot_tsp_solution_animation, create_proba_plot, create_distance_plot
from TSPReader import *
import time

from TraversalPath import TraversalPath

file = "gr202"
filename = f'./dataset/{file}.tsp'
file_solution = f'./dataset/{file}.opt.tour'

# sample testing points
# pr8 = [(10, 10, 1),(25, 15, 2),(40, 10, 3),(45, 25, 4),(40, 40, 5),(45, 45, 6),(10, 40, 7),(5, 25, 8), (20, 30, 9), (30, 30, 10), (20, 20, 11), (30, 20, 12)]

tsp_reader = TSPReader(problem_path=filename, solution_path=file_solution)
TraversalPath._pre_calculate_distance_matrix(tsp_reader.problem)
tsp = TSProblem(tsp_reader.problem)

start_time = time.time()
solution = tsp.run(max_iteration=100000, temperature=1.0, alpha=0.005, beta=5e-2)
end_time = time.time()

print("STIMULATED ANNEALING")
print("Best path found length:", solution.get_path_length())
print(f"The function took {(end_time - start_time):.4f} seconds to run.")

start_time = time.time()
greedy_solution = tsp.generate_greedy_solution()
end_time = time.time()

print("GREEDY ALGORITHM")
print("Best path found length:", greedy_solution.get_path_length())
print(f"The function took {(end_time - start_time):.4f} seconds to run.")

create_proba_plot(tsp)
create_distance_plot(tsp)
plot_tsp_solution_animation(cities=tsp_reader.problem,
                            frames=tsp.stored_results,
                            #other_paths=[{'name': 'Greedy Algorithm', 'cities': greedy_solution.path}],
                            best_actual_path=tsp_reader.solution,
                            max_frames=1500)