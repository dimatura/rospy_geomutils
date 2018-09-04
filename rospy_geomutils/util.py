import numpy as np

__all__ = ['norm_rows',
           'norm_cols',
           'homo_rows',
           'homo_cols',
           'dehomo_rows',
           'dehomo_cols',
           'transform_rows',
           'transform_cols',
           'check_orthogonal',
           'safe_arccos',
           ]

def norm_rows(x):
    """ normalize along last dim (rows if 2D matrix)
    """
    x = np.asanyarray(x)
    norms = np.sqrt(np.sum(x**2, axis=-1))
    return x/norms[..., np.newaxis]


def norm_cols(x):
    """ normalize along first dim (cols if 2D matrix)
    """
    x = np.asanyarray(x)
    norms = np.sqrt(np.sum(x**2, axis=0))
    return x/norms


def homo_rows(x):
    """ inserts 1 in last dimension (rows if 2D matrix)
    """
    x = np.asanyarray(x)
    return np.insert(x, x.shape[-1], 1, axis=-1)


def homo_cols(x):
    """ inserts 1 in first dimension (cols if 2D matrix)
    """
    x = np.asanyarray(x)
    return np.insert(x, x.shape[0], 1, axis=0)


def dehomo_rows(x):
    """ divides by last element of last dimension (rows if 2D matrix)
    """
    x = np.asfarray(x)
    return x[..., :-1] / x[..., np.newaxis, -1]


def dehomo_cols(x):
    """ divides by last element of first dimension (cols if 2D matrix)
    """
    x = np.asfarray(x)
    return x[:-1, ...] / x[np.newaxis, -1, ...]


def transform_rows(M, x):
    """ Transform each row as M*x. Make x homogeneous if M is higher dim by 1.
    """
    if M.shape[0] == M.shape[1] == x.shape[-1] + 1:
        xh = homo_rows(x)
        Mxh = np.dot(M, xh.T).T
        Mx = dehomo_rows(Mxh)
        return Mx
    Mx = np.dot(M, x.T).T
    return Mx


def transform_cols(M, x):
    """ Transform each column as M*x. Make x homogeneous if M is higher dim by 1.
    """
    if M.shape[0] == M.shape[1] == x.shape[0] + 1:
        xh = homo_cols(x)
        Mxh = np.dot(M, xh)
        Mx = dehomo_cols(Mxh)
        return Mx
    Mx = np.dot(M, x)
    return Mx


def check_orthogonal(x):
    """ Check that the argument is an orthogonal matrix.
    """
    N = x.shape[0]
    I = np.eye(N)
    rtol = 10e-10
    atol = 10e-7
    assert(np.allclose(I, np.dot(x, x.T), rtol=rtol, atol=atol))
    assert(np.allclose(I, np.dot(x.T, x), rtol=rtol, atol=atol))


def safe_arccos(x):
    """ acos clipped to (-1, 1)
    """
    return np.arccos(np.clip(x, -1.0, 1.0))
