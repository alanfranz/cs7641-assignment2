import sys

import mlrose
import numpy as np
from cs7641p2.eightqueens import queens_max


def compare_fitness(problem, init_state, max_attempts=100, max_iters=1_000_000):
    for attempts in (1000, 100_000):
        for max_iters in (1000, 1000*1000):
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



if __name__ == "__main__":
    # fitness =
    # peak1
    #print(fitness.evaluate(np.concatenate([np.ones((11,)), np.zeros((89,))])))
    # peak2
    #print(fitness.evaluate(np.concatenate([np.ones((89,)), np.zeros((11,))])))
    # sys.exit(0)
    # Initialize custom fitness function object
    problem = mlrose.DiscreteOpt(length=100, fitness_fn=mlrose.FourPeaks(t_pct=0.10), maximize=True, max_val=2)
    # TODO: prova con array casuale iniziale
    compare_fitness(problem, np.zeros((100, )))
    sys.exit(0)

    init_state = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    fitness_cust = mlrose.CustomFitness(queens_max)
    problem = mlrose.DiscreteOpt(length=8, fitness_fn=fitness_cust, maximize=True, max_val=8)
    print("EIGHT QUEENS:")
    compare_fitness(problem, init_state)
    print("-----------------\n")

    init_state = np.zeros((16, ), dtype=int)
    #fitness_cust = mlrose.CustomFitness(queens_max)
    problem = mlrose.DiscreteOpt(length=8, fitness_fn=fitness_cust, maximize=True, max_val=8)
    print("FOUR PEAKS:")
    compare_fitness(problem, init_state)
    print("-----------------\n")



