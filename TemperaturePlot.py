import matplotlib.pyplot as plt

alpha1 = 0.995
alpha2 = 0.005
def decrease_temperature(temp, iteration) -> float:
    return temp * alpha1


def decrease_temperature2(temp, alpha, iteration, initial_temp) -> float:
    return initial_temp / (1 + (alpha * iteration))


iterations = 10000
initial_temp = 1.0
iteration_values = list(range(iterations))
values = [initial_temp]
values2 = [initial_temp]

for i in range(iterations-1):
    values.append(decrease_temperature2(values[i], alpha2, i, initial_temp))
    values2.append(decrease_temperature(values2[i], i))

plt.plot(iteration_values, values, label=f"Hyperbolic decay (alpha={alpha2})")
plt.plot(iteration_values, values2, label=f"Exponential decay (alpha={alpha1})")
plt.title("Temperature decrease over each iteration")
plt.xlabel("Iteration")
plt.ylabel("Temperature")
plt.legend()
plt.grid(True)
plt.show()