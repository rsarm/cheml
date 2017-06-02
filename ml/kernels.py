import numpy as np





def rbf_kernel(edm,gamma):
    """xxx."""

    return np.exp(-gamma * edm)





def laplacian_kernel(mdm,gamma):
    """xxx."""

    return np.exp(-gamma * mdm)





def multiquadric_cityblock_kernel(mdm,c):
    """xxx."""

    return mdm+np.abs(c)





def multiquadric_euclidean_kernel(edm,c):
    """xxx."""

    return np.sqrt(edm + c**2)






def rational_cityblock_kernel(mdm,c):
    """xxx."""

    return 1.000 - mdm / (mdm + np.abs(c))






def rational_euclidean_kernel(edm,c):
    """xxx."""

    return 1.000 - edm / (edm + c**2)






def inv_multiquadric_cityblock_kernel(mdm,c):
    """xxx."""

    return 1. / ( mdm + np.abs(c) )






def inv_multiquadric_euclidean_kernel(edm,c):
    """xxx."""

    return 1. / np.sqrt(edm + c**2)






def spherical_cityblock_kernel(mdm,gamma):
    """xxx."""


    cutoff=(mdm < 1./gamma)*1.

    return cutoff*(1 - 1.50*gamma*mdm + 0.50*(gamma*mdm)**3)






def spherical_euclidean_kernel(edm,gamma):
    """xxx."""

    cutoff=(edm < 1./gamma)*1.

    return cutoff*(1 - 1.50*gamma*edm + 0.50*(gamma*edm)**3)











