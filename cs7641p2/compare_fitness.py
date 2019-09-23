import mlrose
import numpy as np
from cs7641p2.eightqueens import queens_max


def compare_fitness(problem, init_state, max_attempts=100, max_iters=1000):
    # Solve problem using simulated annealing and exponential decay
    best_state, best_fitness, fitness_curve = mlrose.simulated_annealing(problem, schedule=mlrose.decay.ExpDecay(),
                                                                         max_attempts=max_attempts, max_iters=max_iters,
                                                                         init_state=init_state, random_state=1,
                                                                         curve=True)

    print(best_state)

    print(best_fitness)

    best_state, best_fitness, fitness_curve = mlrose.random_hill_climb(problem, max_attempts=max_attempts,
                                                                       max_iters=max_iters,
                                                                       init_state=init_state, random_state=1,
                                                                       curve=True)

    print(best_state)
    print(best_fitness)

    best_state, best_fitness, fitness_curve = mlrose.genetic_alg(problem, max_attempts=max_attempts,
                                                                 max_iters=max_iters, curve=True, random_state=1)

    print(best_state)
    print(best_fitness)

    best_state, best_fitness, fitness_curve = mlrose.mimic(problem, max_attempts=max_attempts, max_iters=max_iters,
                                                           curve=True, random_state=1)

    print(best_state)
    print(best_fitness)


if __name__ == "__main__":
    # Initialize custom fitness function object
    init_state = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    fitness_cust = mlrose.CustomFitness(queens_max)
    problem = mlrose.DiscreteOpt(length=8, fitness_fn=fitness_cust, maximize=True, max_val=8)

    compare_fitness(problem, init_state)
