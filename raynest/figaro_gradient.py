try:
    from figaro.mixture import DPGMM
    from figaro.likelihood import log_norm
except:
    print("FIGARO not available! Hamiltonian Monte Carlo sampling not supported!")
    pass

import numpy as np
from tqdm import tqdm
from scipy.stats import multivariate_normal as mn

class ADPGMM(DPGMM):
    def __init__(self, *args, **kwargs):
        super(ADPGMM, self).__init__(*args, **kwargs)
        
    def _pdf_array(self, x):
        """
        Evaluate mixture at point(s) x in probit space
        
        Arguments:
            :np.ndarray x: point(s) to evaluate the mixture at (in probit space)
        
        Returns:
            :np.ndarray: comp.pdf(x)
        """
        return np.array([w*np.exp(log_norm(x[0], comp.mu, comp.sigma)) for comp, w in zip(self.mixture, self.w)])

    def _log_gradient(self, x):
        """
        Returns the log gradient of the mixture
        
        Arguments:
            :np.ndarray x: point to evaluate the gradient at
        
        Returns:
            :np.ndarray: log_gradient
        """
        p = self._pdf_array(x)
        B = np.array([-np.dot(np.linalg.inv(comp.sigma),(x - comp.mu)) for comp in self.mixture])

        return np.average(B, weights = p, axis = 0)

if __name__ == "__main__":
    dim = 1
    N = 10000
    data = np.random.normal(0.0,1.0, size = (N, dim))

    mix = ADPGMM([[-10, 10]], probit = False)
    for s in tqdm(data):
        mix.add_new_point(s)

    mix.build_mixture()
    print(mix._log_gradient([0.0]))

    import matplotlib.pyplot as plt

    x = np.linspace(-10,10,101)
    G = [mix._log_gradient(xi) for xi in x]
    f = plt.figure()
    ax = f.add_subplot(121)
    ax.plot(x,mix.pdf(x))
    ax.hist(data,density=True,bins=100)
    ax = f.add_subplot(122)
    ax.plot(x,G,color='r')
    ax.plot(x,np.gradient(mix.logpdf(x),np.diff(x)[0]),color='k')
    plt.show()





