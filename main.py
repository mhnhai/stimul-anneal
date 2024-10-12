from TSProblem import TSProblem, plot_tsp_solution, plot_tsp_solution_animation
from CityGenerator import *
from TSPReader import *
import time

filename = './dataset/att48.tsp'
file_solution = './dataset/att48.opt.tour'

start_time = time.time()
tsp_reader = TSPReader(problem_path=filename, solution_path=file_solution)
tsp = TSProblem(tsp_reader.problem)
best_path, best_length = tsp.run(max_iteration=100000, temperature=1.0, alpha=0.005, beta=0.01, greedy_solution=False)
end_time = time.time()
print("Best path found:", best_path)
print("Best path found length:", best_length)
print(f"Execution time: {end_time - start_time} seconds")

#plot_tsp_solution(cities=tsp_reader.problem, best_path=best_path, best_actual_path=tsp_reader.solution)
plot_tsp_solution_animation(cities=tsp_reader.problem, frames=tsp.stored_results, best_actual_path=tsp_reader.solution, max_frames=500)