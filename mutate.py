import numpy as np


def add(x, i, delta):
    x[i] = x[i] + delta
    return x

def sub(x, i, delta):
    x[i] = x[i] - delta
    return x    

def enumerate_mutants(x, xmin=-1, xmax=1, delta=0.01):
    """
    Given parent response x, return all possible mutants (difference encoded).
    
    Returns matrix M such that each row of (x + M) is one mutant response, each
    with the same support as x.
    
    >>> enumerate_mutants(np.array([-0.99, 0.99, 0, -1, 1]))

	array([[ 0.01,  0.  ,  0.  ,  0.  ,  0.  ],
		   [ 0.  ,  0.01,  0.  ,  0.  ,  0.  ],
		   [ 0.  ,  0.  ,  0.  ,  0.01,  0.  ],
		   [-0.01,  0.  ,  0.  ,  0.  ,  0.  ],
		   [ 0.  , -0.01,  0.  ,  0.  ,  0.  ],
		   [ 0.  ,  0.  ,  0.  ,  0.  , -0.01],
		   [ 0.99,  0.  ,  0.01,  0.  ,  0.  ],
		   [ 0.99,  0.  , -0.01,  0.  ,  0.  ],
		   [ 0.  , -0.99,  0.01,  0.  ,  0.  ],
		   [ 0.  , -0.99, -0.01,  0.  ,  0.  ],
		   [ 0.  ,  0.  ,  0.01,  1.  ,  0.  ],
		   [ 0.  ,  0.  , -0.01,  1.  ,  0.  ],
		   [ 0.  ,  0.  ,  0.01,  0.  , -1.  ],
		   [ 0.  ,  0.  , -0.01,  0.  , -1.  ]])
    
    """
    m = len(x)
    i = (x == 0)
    j = np.invert(i)
    
    # Each row of M adds +delta or -delta to one non-zero component.
    M = []
    M += [add(np.zeros(m), k, delta) for k in np.nonzero(j & (x < xmax))[0]]
    M += [sub(np.zeros(m), k, delta) for k in np.nonzero(j & (x > xmin))[0]]
    M = np.array(M)
    
    # Each row of N puts +delta or -delta on on of the zero components.
    N = []
    N += [add(np.zeros(m), k, delta) for k in np.nonzero(i)[0]]
    N += [sub(np.zeros(m), k, delta) for k in np.nonzero(i)[0]]
    N = np.array(N)
    print N
    
    # Each S swaps out non-zero component k and adds new support according to N.
    for k in np.nonzero(j)[0]:
        S = N.copy()
        S[:, k] = -x[k]
        M = np.concatenate([M, S])
    
    return M