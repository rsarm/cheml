import numpy as np

from scipy.optimize import minimize,basinhopping

from cheml.ml import kernels





class krr(object):
    """Kernel Ridge Regressor."""

    def __init__(self,kernel,gamma,alpha=1e-10):
        self.kernel = kernel.lower()
        self.gamma  = gamma
        self.alpha  = alpha

        implemented_kernels=['rbf','gaussian',
                             'laplacian']

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























































