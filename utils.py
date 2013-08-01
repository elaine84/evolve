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
    return np.matrix(np.diag(np.random.uniform(low=low, high=high, size=n)))

def random_rotation(n):
    """
    Generate a random rotation matrix.

    Adapted from:  http://arxiv.org/pdf/math-ph/0609050v2.pdf and
    http://www.mathworks.com/matlabcentral/newsreader/view_thread/298500

    **Parameters**

        **n** : int

            Size of the rotation matrix.

    **Returns**

        **q** : `numpy.matrix`

            Random rotation matrix.

    """
    z = np.random.randn(n, n)
    (q, r) = np.linalg.qr(z)
    if (np.linalg.det(q) < 0):
        i = int(np.random.random() * n)
        q[:, i] = -q[:, i]
    assert (np.abs(np.linalg.inv(q) - q.T) < 10**(-12)).all(), np.linalg.inv(q)
    assert np.abs(np.linalg.det(q) - 1.0) < 10**(-12), np.linalg.det(q)
    return np.matrix(q)
