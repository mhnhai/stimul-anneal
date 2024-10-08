from TSProblem import TSProblem, plot_tsp_solution
from CityGenerator import *
from TSPReader import *

filename = './dataset/a280.tsp'
file_solution = './dataset/a280.opt.tour'

tsp_reader = TSPReader(problem_path=filename, solution_path=file_solution)
tsp = TSProblem(tsp_reader.problem)
best_path, best_length = tsp.run(max_iteration=1000, temperature=0.9)
print("Best path:", best_path)
print("Best path length:", best_length)

plot_tsp_solution(cities=tsp_reader.problem, best_path=best_path, best_actual_path=tsp_reader.solution)
