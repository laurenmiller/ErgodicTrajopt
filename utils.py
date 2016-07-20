import numpy as np

def matmult(*x):
    """
    Shortcut for standard matrix multiplication.
    matmult(A,B,C) returns A*B*C.
    """
    return reduce(np.dot, x)


def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like 1-D arrays to form the cartesian product of.
    out : ndarray Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))"""

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out

def plot_all(X):
    # plot a single trace
    tj=X.T
    for ln in range(0,tj.shape[0]):
        plt.plot(tj[ln])
    plt.show()
    
class pdf(object):
    def __init__(self, res = 400, wlimit=float(1), dimw=2):
        self.dimw = dimw # workspace dimension. current implementation only works for 2D
        self.dim=dim
        self.wlimit = wlimit
        self.res = res

def uniform_pdf(X):
    # define a uniform PDF over grid X
    return np.ones(X[0].shape)
        

def bimodal_gaussian_pdf(X, mean=[[.25,.3],[.24,.4]], covar=[.1,.1]):
    # define a Gaussian PDF over grid X
    
    pd= np.exp(-((X[0] - mean[0][0])**2/( 2*covar[0]**2)) -
               ((X[1] - mean[0][1])**2/(2*covar[1]**2))) + \
               np.exp(-((X[0] - mean[1][0])**2/( 2*covar[0]**2)) -
                      ((X[1] - mean[1][1])**2/(2*covar[1]**2)))
    return pd

