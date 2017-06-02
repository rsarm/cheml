import numpy as np

from scipy.optimize import minimize,basinhopping



def grid_search(mae, gamma_range, dmtr, dmte, ytr, yte):
    """xxx."""

    err=1.e5
    gamma=[]

    for g in gamma_range:
        err_g = mae(g,dmtr,dmte,ytr, yte)

        if  err_g < err:
            gamma = np.copy(g)
            err   = np.copy(err_g)

    return gamma,err





def simplex(mae, gamma0, dmtr, dmte, ytr, yte, maxiter, tol):
    """xxx."""

    res=minimize(mae,
                 gamma0,
                 args    = (dmtr, dmte, ytr, yte),
                 method  = 'Nelder-Mead',
                 tol     = tol,
                 options = {'maxiter':maxiter}
                )

    return res.x[0], res.fun




#func_args={"args"  :(xtr,ytr,xte,yte),
#           "method":'Nelder-Mead'}
#res=basinhopping(self._mae,gamma0,minimizer_kwargs=func_args,
#         #accept_test=mybounds,
#         #callback=print_fun,
#         niter=niter
#        )


