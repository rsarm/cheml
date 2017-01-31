import numpy as np


from cheml.tools.metrics  import euclidean2, manhattan



def rbf_kernel(x,y,gamma):
    """xxx."""

    return np.exp(-gamma*euclidean2(x,y))





def laplacian_kernel(x,y,gamma):
    """xxx."""

    return np.exp(-gamma*manhattan(x,y))





def multiquadric_cityblock_kernel(x,y,c):
    """xxx."""

    return manhattan(x,y)+np.abs(c)





def multiquadric_euclidean_kernel(x,y,c):
    """xxx."""

    return np.sqrt(euclidean2(x,y)+c**2)






def rational_cityblock_kernel(x,y,c):
    """xxx."""

    mdm=manhattan(x,y)

    return 1.000 - mdm / (mdm + np.abs(c))






def rational_euclidean_kernel(x,y,c):
    """xxx."""

    edm=euclidean2(x,y)

    return 1.000 - edm / (edm + c**2)
