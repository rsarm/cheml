import numpy as np

from scipy.optimize import minimize,basinhopping



def grid_search(f, gamma_range, xtr, ytr, xte, yte):
    """xxx."""

    err=1.e5
    gamma=[]

    for g in gamma_range:
        err_g = f(g,xtr,ytr,xte,yte)

        if  err_g < err:
            gamma = np.copy(g)
            err   = np.copy(err_g)

    return gamma,err





def simplex(f, gamma0, xtr, ytr, xte, yte, tol=1.e-5):
    """xxx."""

    res=minimize(f,
                 gamma0,
                 args   = (xtr,ytr,xte,yte),
                 method = 'Nelder-Mead',
                 tol    = tol
                )

    return res.x[0], res.fun


#func_args={"args"  :(xtr,ytr,xte,yte),
#           "method":'Nelder-Mead'}
#res=basinhopping(self._mae,gamma0,minimizer_kwargs=func_args,
#         #accept_test=mybounds,
#         #callback=print_fun,
#         niter=niter
#        )


