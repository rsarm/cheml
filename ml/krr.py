import numpy as np

from cheml.ml import kernels
from cheml.ml.optimization import grid_search, simplex
from cheml.tools.metrics  import euclidean2, manhattan







class krr(object):
    """Kernel Ridge Regressor."""

    def __init__(self,kernel,alpha=1e-10,**kwargs):#gamma=1e-5,c=1e-5):
        self.kernel = kernel.lower()
        self.alpha  = alpha


        implemented_kernels=['rbf','gaussian',
                             'laplacian',
                             'multiquadric_euclidean','mql2',
                             'multiquadric_cityblock','mql1',
                             'inv_multiquadric_euclidean','imql2',
                             'inv_multiquadric_cityblock','imql1',
                             'rational_euclidean','ratl2',
                             'rational_cityblock','ratl1',
                             'spherical_cityblock','sphl1',
                             'spherical_euclidean','sphl2']

        if self.kernel not in implemented_kernels:
            raise ValueError("Kernel \'"+self.kernel+"\' not implemented.")
                  # or NotImplementedError


        if self.kernel=='rbf' or self.kernel=='gaussian':
            self._kernel = kernels.rbf_kernel
            self.kparams = parse_params('gamma',kwargs)
            self._dismat = euclidean2

        if self.kernel=='laplacian':
            self._kernel = kernels.laplacian_kernel
            self.kparams = parse_params('gamma',kwargs)
            self._dismat = manhattan

        if self.kernel=='multiquadric_euclidean' or self.kernel=='mql2':
            self._kernel = kernels.multiquadric_euclidean_kernel
            self.kparams = parse_params('c',kwargs)
            self._dismat = euclidean2

        if self.kernel=='multiquadric_cityblock' or self.kernel=='mql1':
            self._kernel = kernels.multiquadric_cityblock_kernel
            self.kparams = parse_params('c',kwargs)
            self._dismat = manhattan

        if self.kernel=='inv_multiquadric_euclidean' or self.kernel=='imql2':
            self._kernel = kernels.inv_multiquadric_euclidean_kernel
            self.kparams = parse_params('c',kwargs)
            self._dismat = euclidean2

        if self.kernel=='inv_multiquadric_cityblock' or self.kernel=='imql1':
            self._kernel = kernels.inv_multiquadric_cityblock_kernel
            self.kparams = parse_params('c',kwargs)
            self._dismat = manhattan

        if self.kernel=='rational_euclidean' or self.kernel=='ratl2':
            self._kernel = kernels.rational_euclidean_kernel
            self.kparams = parse_params('c',kwargs)
            self._dismat = euclidean2

        if self.kernel=='rational_cityblock' or self.kernel=='ratl1':
            self._kernel = kernels.rational_cityblock_kernel
            self.kparams = parse_params('c',kwargs)
            self._dismat = manhattan

        if self.kernel=='spherical_cityblock' or self.kernel=='sphl1':
            self._kernel = kernels.spherical_cityblock_kernel
            self.kparams = parse_params('gamma',kwargs)
            self._dismat = manhattan

        if self.kernel=='spherical_euclidean' or self.kernel=='sphl2':
            self._kernel = kernels.spherical_euclidean_kernel
            self.kparams = parse_params('gamma',kwargs)
            self._dismat = euclidean2





    def __repr__(self):

        k= "kernel=\'"  + self.kernel+"\'"
        a= ", alpha=" + str(self.alpha)

        params_str=[pi +'='+ str(self.kparams[pi]) for pi in self.kparams]

        return "krr(" + k + a + ', ' + ', '.join(params_str) + ")"






    def _fit(self,dmtr,ytr):
        """Find the regression coefficients and intializes
        the variable self.coeff.

        The fit is splitted in two functions, because self._fit
        is needed by self.optimize_kernel and self.fit is
        done to have the same structure as sklearn.
        """

        ktr=self._kernel(dmtr,**self.kparams)

        self.coeff=np.linalg.solve(ktr+self.alpha*np.eye(ktr.shape[0]),ytr)






    def fit(self,xtr,ytr):
        """Find the regression coefficients and intializes
        the variable self.coeff.

        Version of self._fit for the user.
        """

        self.xtr=xtr # To be used by self.predict without passing it as argument.

        dmtr = self._dismat(xtr,xtr)

        self._fit(dmtr,ytr)






    def _predict(self,dmte):
        """Returns the predicted function values for the
        data points in xte.

        The prediction is splitted in two functions, because self._predict
        is needed by self.optimize_kernel and self.predict is
        done to have the same structure as sklearn.
        """

        kte=self._kernel(dmte,**self.kparams)

        return np.dot(kte,self.coeff)






    def predict(self,xte):
        """Returns the predicted function values for the
        data points in xte.

        Version of self._predict for the user.
        """

        dmte = self._dismat(xte,self.xtr)

        return self._predict(dmte)







    def _mae(self,gamma,dmtr,dmte,ytr,yte):
        """xxx."""

        self.kparams['gamma']=np.abs(gamma)

        self._fit(dmtr,ytr)

        yp=self._predict(dmte)

        return np.abs(yte-yp).sum()/yp.shape[0]







    def optimize_kernel(self,xtr,ytr,xte,yte,gamma0=1e-5,
                        gamma_range=[1.e-5],optmod='grid_search', maxiter=100, tol=1.e-5):
        """Run an optimization to find a
        minimum of the MAE in function of the kernel width
        gamma.

        Returns the lowest error and updates kparams['gamma'].
        """

        dmtr = self._dismat(xtr,xtr)
        dmte = self._dismat(xte,xtr)

        if optmod=='simplex':
            gamma,err = simplex(    self._mae, gamma0,      dmtr, dmte, ytr, yte, maxiter, tol)

        if optmod=='grid_search':
            gamma,err = grid_search(self._mae, gamma_range, dmtr, dmte, ytr, yte)

        self.kparams['gamma']=gamma

        return err






def parse_params(param,dict_param):
    """xxx."""

    if type(param)!=list:
        param=[param]

    arg_list=[k for k in dict_param]

    _params={}

    for pi in param:
        if pi in arg_list:
            _params[pi]=dict_param[pi]
        else:
            raise ValueError("Wrong parameter for the selected kernel.")

    return _params

















































