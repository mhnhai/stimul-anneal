from TSProblem import TSProblem, plot_tsp_solution_animation
from TSPReader import *

file = "att48"
filename = f'./dataset/{file}.tsp'
file_solution = f'./dataset/{file}.opt.tour'

tsp_reader = TSPReader(problem_path=filename, solution_path=file_solution)
tsp = TSProblem(tsp_reader.problem)
best_path, best_length = tsp.run(max_iteration=100000, temperature=1.0, alpha=0.005, beta=0.01, greedy_solution=False)
print("Best path found:", best_path)
print("Best path found length:", best_length)

#plot_tsp_solution(cities=tsp_reader.problem, best_path=best_path, best_actual_path=tsp_reader.solution)
plot_tsp_solution_animation(cities=tsp_reader.problem, frames=tsp.stored_results, best_actual_path=tsp_reader.solution, max_frames=1500)
