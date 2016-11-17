import numpy as np

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

        if self.kernel=='laplacian':
            try:
                from sklearn.metrics.pairwise import manhattan_distances
            except:
                raise ValueError("The Laplacian kernel needs sklearn.")





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






