class TSPReader:
    def __init__(self, problem_path, solution_path):
        self.problem_path = problem_path
        self.solution_path = solution_path
        self.problem = self.read_tsp_file(problem_path)
        self.solution = self.read_tsp_solution_file(solution_path)

        print("TS PROBLEM INFORMATION")
        print("CITIES")
        print(self.problem)
        print(self.solution)
    def read_tsp_file(self, filename):
        with open(filename, 'r') as file:
            lines = file.readlines()

        coordinates = []
        reading_coordinates = False

        for line in lines:
            line = line.strip()
            if line.startswith("NODE_COORD_SECTION"):
                reading_coordinates = True
                continue
            elif line == "EOF":
                break

            if reading_coordinates:
                parts = line.split()
                if len(parts) == 3:
                    coordinates.append((float(parts[1]), float(parts[2])))

        return coordinates

    def read_tsp_solution_file(self, filename):
        with open(filename, 'r') as file:
            lines = file.readlines()

        tour = []
        reading_tour = False

        for line in lines:
            line = line.strip()
            if line.startswith("TOUR_SECTION"):
                reading_tour = True
                continue
            elif line == "EOF":
                break

            if reading_tour:
                try:
                    node = int(line)
                    if node != -1:
                        tour.append(node-1)
                except ValueError:
                    continue

        return tour
