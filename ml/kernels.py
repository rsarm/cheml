import numpy as np

from cheml.descriptors.base import euclidean

# The laplacian kernel currently depends on the manhattan
# distance implemented in sklean.
try:
    from sklearn.metrics.pairwise import manhattan_distances
except:
    pass




def rbf_kernel(x,y,gamma):
    """xxx."""

    return np.exp(-gamma*euclidean(x,y))



def laplacian_kernel(x,y,gamma):
    """xxx."""

    return np.exp(-gamma*manhattan_distances(x,y))


