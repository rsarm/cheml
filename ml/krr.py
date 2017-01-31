import numpy as np

from scipy.optimize import minimize,basinhopping

from cheml.ml import kernels





class krr(object):
    """Kernel Ridge Regressor."""

    def __init__(self,kernel,alpha=1e-10,gamma=1e-5,c=1e-5):
        self.kernel = kernel.lower()
        self.gamma  = gamma
        self.alpha  = alpha
        self.c      = c

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




    def __repr__(self):

        k= "kernel=\'"  + self.kernel+"\'"
        g= ", gamma=" + str(self.gamma)
        a= ", alpha=" + str(self.alpha)

        return "krr(" + k + g + a + ")"






    def _kernel(self,xi,xj):
        """xxx."""

        if self.kernel=='rbf' or self.kernel=='gaussian':
            return kernels.rbf_kernel(xi,xj,self.gamma)

        if self.kernel=='laplacian':
            return kernels.laplacian_kernel(xi,xj,self.gamma)

        if self.kernel=='multiquadric_euclidean' or self.kernel=='mql2':
            return kernels.multiquadric_euclidean_kernel(xi,xj,self.c)

        if self.kernel=='multiquadric_cityblock' or self.kernel=='mql1':
            return kernels.multiquadric_cityblock_kernel(xi,xj,self.c)

        if self.kernel=='inv_multiquadric_euclidean' or self.kernel=='imql2':
            return kernels.inv_multiquadric_euclidean_kernel(xi,xj,self.c)

        if self.kernel=='inv_multiquadric_cityblock' or self.kernel=='imql1':
            return kernels.inv_multiquadric_cityblock_kernel(xi,xj,self.c)

        if self.kernel=='rational_euclidean' or self.kernel=='ratl2':
            return kernels.rational_euclidean_kernel(xi,xj,self.c)

        if self.kernel=='rational_cityblock' or self.kernel=='ratl1':
            return kernels.rational_cityblock_kernel(xi,xj,self.c)

        if self.kernel=='spherical_cityblock' or self.kernel=='sphl1':
            return kernels.spherical_cityblock_kernel(xi,xj,self.gamma)

        if self.kernel=='spherical_euclidean' or self.kernel=='sphl2':
            return kernels.spherical_euclidean_kernel(xi,xj,self.gamma)

    def _get_ktr(self,xtr):
        """Return the kernel of a dataset to itself."""

        return self._kernel(xtr,xtr)




    def _get_kte(self,xte,xtr):
        """Returns the kernel between two datasets."""

        return self._kernel(xte,xtr)




    def fit(self,xtr,ytr):
        """Find the regression coefficients and intializes
        the variable self.coeff.
        """

        self.xtr=xtr

        self.ktr=self._get_ktr(self.xtr)

        self.coeff=np.linalg.solve(self.ktr+self.alpha*np.eye(xtr.shape[0]),ytr)




    def predict(self,xte):
        """Returns the predicted function values for the
        data points in xte.
        """

        self.kte=self._get_kte(xte,self.xtr)

        return np.dot(self.kte,self.coeff)





    def _mae(self,gamma,xtr,ytr,xte,yte):
        """xxx."""

        self.gamma=np.abs(gamma)

        self.fit(xtr,ytr)

        yp=self.predict(xte)

        return np.abs(yte-yp).sum()/yp.shape[0]






    def optimize_kernel(self,xtr,ytr,xte,yte,gamma0=1e-5,niter=100):
        """Run a Basin Hopping optimization to find local
        minima of the MAE in function of the kernel width
        gamma

        Returns the lowest minima.
        ."""

        func_args={"args"  :(xtr,ytr,xte,yte),
                   "method":'Nelder-Mead'}

        res=basinhopping(self._mae,gamma0,minimizer_kwargs=func_args,
                 #accept_test=mybounds,
                 #callback=print_fun,
                 niter=niter
                )
        print res.fun

        self.gamma=np.abs(res.x[0])

        return res.x























































