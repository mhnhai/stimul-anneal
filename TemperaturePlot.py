import matplotlib.pyplot as plt


def decrease_temperature(temp, iteration) -> float:
    return temp * 0.9995


def decrease_temperature2(temp, alpha, iteration, initial_temp) -> float:
    return initial_temp / (1 + (alpha * iteration))


iterations = 10000
initial_temp = 0.7
iteration_values = list(range(iterations))
values = [initial_temp]

for i in range(iterations-1):
    values.append(decrease_temperature2(values[i], 0.01, i, initial_temp))

plt.plot(iteration_values, values)
plt.title("Temperature decrease over each iteration")
plt.xlabel("Iteration")
plt.ylabel("Temperature")
plt.grid(True)
plt.show()