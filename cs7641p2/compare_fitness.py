import sys

import mlrose
import numpy as np
from cs7641p2.eightqueens import queens_max
from functools import partial


def find_best_mimic(problem, init_state, max_attempts=100, max_iters=1_000_000):
    for attempts in (1000, 100_000):
        for max_iters in (1000, 1000 * 1000):
            for pop_size in (10, 10000):
                best_state, best_fitness, fitness_curve = mlrose.mimic(problem,
                                                                       max_attempts=attempts,
                                                                       pop_size=pop_size,
                                                                       max_iters=max_iters,
                                                                       curve=True,
                                                                       random_state=1)
                print(f"{attempts} / {max_iters} / {pop_size}")
                print(best_state)
                print(best_fitness)


def calc_average_fitness(callable, *args, **kwargs):
    RANDOMS = [2182823240, 220395238, 941243409, 942612051, 2109339754, 1817228411, 1202710073, 2064866518, 1764906721,
               793837664]
    v = 0
    for x in RANDOMS:
        np.random.seed(x)
        state, fitness = callable(*args, **kwargs)
        v += fitness

    return v / 10


def compare_fitness(problem, init_state, max_attempts=100, max_iters=500_000, randoms=()):
    out = {}

    # Solve problem using simulated annealing and exponential decay
    avgfitness = calc_average_fitness(mlrose.simulated_annealing, problem, schedule=mlrose.decay.ExpDecay(),
                                      max_attempts=max_attempts,
                                      max_iters=max_iters,
                                      init_state=init_state,
                                      curve=False)

    print(f"annealing: {avgfitness}")

    avgfitness = calc_average_fitness(mlrose.random_hill_climb, problem, max_attempts=max_attempts,
                                      max_iters=max_iters,
                                      init_state=init_state)

    print(f"random hill climbing: {avgfitness}")
    avgfitness = calc_average_fitness(mlrose.genetic_alg, problem, max_attempts=max_attempts, max_iters=max_iters)
    print(f"genetic: {avgfitness}")

    avgfitness = calc_average_fitness(mlrose.mimic, problem,
                                      max_attempts=max_attempts,
                                      pop_size=200,
                                      max_iters=max_iters)
    print(f"mimic: {avgfitness}")


if __name__ == "__main__":
    np.random.seed(2182823240)
    full_array = np.random.randint(0, 2, 100)

    print("four peaks, T 10%, 10-elements array")
    problem = mlrose.DiscreteOpt(length=10, fitness_fn=mlrose.FourPeaks(t_pct=0.10), maximize=True, max_val=2)
    compare_fitness(problem, full_array[0:10, ])

    print("four peaks, T 10%, 20-elements array")
    problem = mlrose.DiscreteOpt(length=20, fitness_fn=mlrose.FourPeaks(t_pct=0.10), maximize=True, max_val=2)
    compare_fitness(problem, full_array[0:20, ])

    print("four peaks, T 10%, 50-elements array")
    problem = mlrose.DiscreteOpt(length=50, fitness_fn=mlrose.FourPeaks(t_pct=0.10), maximize=True, max_val=2)
    compare_fitness(problem, full_array[0:50])



    sys.exit(0)

    init_state = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    fitness_cust = mlrose.CustomFitness(queens_max)
    problem = mlrose.DiscreteOpt(length=8, fitness_fn=fitness_cust, maximize=True, max_val=8)
    print("EIGHT QUEENS:")
    compare_fitness(problem, init_state)
    print("-----------------\n")

    init_state = np.zeros((16,), dtype=int)
    # fitness_cust = mlrose.CustomFitness(queens_max)
    problem = mlrose.DiscreteOpt(length=8, fitness_fn=fitness_cust, maximize=True, max_val=8)
    print("FOUR PEAKS:")
    compare_fitness(problem, init_state)
    print("-----------------\n")
