import numpy as np


def random_correlation(n, low=0.1, high=1.0):
    """
    Generates a random correlation matrix for an axis-aligned n-dim Gaussian.
    
    **Parameters**
    
        **n** : int
        
            Size of the correlation matrix.
        
        **low** : float, optional
        
            Minimum possible correlation value, not inclusive.  The default 
            value is 0.1.
        
        **high** : float, optional
        
            Maximum possible correlation value, not inclusive.  The default 
            value is 1.0.
    
    **Returns**
    
        **cor** : `numpy.matrix`
        
            Random correlation matrix for an axis-aligned n-dim Gaussian, so it
            is a diagonal (n x n) matrix.  Each entry is ~ U(0.1, 1).
    
    """
    return np.diag(np.random.uniform(low=low, high=high, size=n))

