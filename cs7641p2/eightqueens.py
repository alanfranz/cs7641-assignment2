# SOURCE: https://mlrose.readthedocs.io/en/stable/source/tutorial1.html
import mlrose
import numpy as np

def queens_max(state):
    # Define alternative N-Queens fitness function for maximization problem
    # Initialize counter
    fitness_cnt = 0

    # For all pairs of queens
    for i in range(len(state) - 1):
        for j in range(i + 1, len(state)):
            # Check for horizontal, diagonal-up and diagonal-down attacks
            if (state[j] != state[i]) \
                    and (state[j] != state[i] + (j - i)) \
                    and (state[j] != state[i] - (j - i)):
                # If no attacks, then increment counter
                fitness_cnt += 1

    return fitness_cnt

if __name__ == "__main__":
    # Initialize custom fitness function object
    fitness_cust = mlrose.CustomFitness(queens_max)
    fitness = mlrose.Queens()
    problem = mlrose.DiscreteOpt(length=8, fitness_fn=fitness, maximize=False, max_val=8)

    schedule = mlrose.ExpDecay()

    # Define initial state
    init_state = np.array([0, 1, 2, 3, 4, 5, 6, 7])

    # Solve problem using simulated annealing
    best_state, best_fitness = mlrose.simulated_annealing(problem, schedule=schedule,
                                                          max_attempts=100, max_iters=1000,
                                                          init_state=init_state, random_state=1)

    print(best_state)

    print(best_fitness)

