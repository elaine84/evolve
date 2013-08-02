import numpy as np

import error
import mutate
import utils


def evolve(k=10, n=100, low=0.1, high=1.0, xmin=-1, xmax=1, delta=0.01, T=20):

    # Generate random target response.
    w = utils.random_response(k=k, n=n)

    # Generate random initial response.
    u = utils.random_response(k=k, n=n)

    # Generate random rotation matrix.
    rotation = utils.random_rotation(n)

    # Generate random correlation matrix.
    cor = utils.random_correlation(n, low=low, high=high)

    # Expected L2 loss of the initial response.
    se = error.sq_err(w, u, rotation, cor)
    print se

    for t in range(T):

        # Generate all possible mutants.
        # Um this forgot to apply the diffs...
        M = mutate.enumerate_mutants(np.array(u).flatten(), xmin=xmin, xmax=xmax, delta=delta).T

        # Expected L2 loss for each of the mutant responses.
        SE = np.array([error.sq_err(w, M[:, i], rotation, cor) for i in range(M.shape[1])])

        i = SE.argmin()

        print SE[i]
        if (SE[i] >= se):
            break

        # Find the winner.
        u = M[:, i]
        se = SE[i]

    return (w, u, rotation, cor)
