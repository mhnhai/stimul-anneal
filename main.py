from TSProblem import TSProblem, plot_tsp_solution_animation, create_proba_plot, create_distance_plot
from TSPReader import *
import time

file = "a280"
filename = f'./dataset/{file}.tsp'
file_solution = f'./dataset/{file}.opt.tour'

tsp_reader = TSPReader(problem_path=filename, solution_path=file_solution)
tsp = TSProblem(tsp_reader.problem)

start_time = time.time()
best_path, best_length = tsp.run(max_iteration=50000, temperature=1.0, alpha=0.005, beta=5e-6, greedy_solution=False)
end_time = time.time()

print("Best path found:", best_path)
print("Best path found length:", best_length)
print(f"The function took {(end_time - start_time):.4f} seconds to run.")

create_proba_plot(tsp)
create_distance_plot(tsp)
plot_tsp_solution_animation(cities=tsp_reader.problem, frames=tsp.stored_results, best_actual_path=tsp_reader.solution, max_frames=1200)
