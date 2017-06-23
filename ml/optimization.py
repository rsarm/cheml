import numpy as np

from scipy.optimize import minimize,basinhopping



def grid_search(mae, param_range, dmtr, dmte, ytr, yte):
    """xxx."""

    err=1.e5
    param=[]

    for g in param_range:
        err_g = mae(g,dmtr,dmte,ytr, yte)

        if  err_g < err:
            param = np.copy(g)
            err   = np.copy(err_g)

    return float(param),err





def simplex(mae, param0, dmtr, dmte, ytr, yte, maxiter, tol):
    """xxx."""

    res=minimize(mae,
                 param0,
                 args    = (dmtr, dmte, ytr, yte),
                 method  = 'Nelder-Mead',
                 tol     = tol,
                 options = {'maxiter':maxiter}
                )

    return res.x[0], res.fun



