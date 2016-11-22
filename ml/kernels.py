import numpy as np

from cheml.descriptors.base import euclidean2, manhattan




def rbf_kernel(x,y,gamma):
    """xxx."""

    return np.exp(-gamma*euclidean2(x,y))



def laplacian_kernel(x,y,gamma):
    """xxx."""

    return np.exp(-gamma*manhattan(x,y))
