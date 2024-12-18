import unittest
import numpy as np
import raynest.model
from scipy import stats

class GaussianModel(raynest.model.Model):
    """
    An n-dimensional gaussian
    """
    def __init__(self,dim=50):
        self.distr = stats.norm(loc=0,scale=1.0)
        self.dim=dim
        self.names=['{0}'.format(i) for i in range(self.dim)]
        self.bounds=[[-10,10] for _ in range(self.dim)]
#        self.bounds[0] = [-50, 50]
        self.analytic_log_Z=0.0 - sum([np.log(self.bounds[i][1]-self.bounds[i][0]) for i in range(self.dim)])

    def log_likelihood(self,p):
        return np.sum([-0.5*p[n]**2-0.5*np.log(2.0*np.pi) for n in p.names])##np.sum([self.distr.logpdf(p[n]
    
    def log_prior(self,p):
        logP = super(GaussianModel,self).log_prior(p)
#        for n in p.names: logP += -0.5*(p[n])**2
        return logP
    
    def force(self, p):
        f = np.zeros(1, dtype = {'names':p.names, 'formats':['f8' for _ in p.names]})
        return f
    
    def analytical_gradient(self, p):
        return p.values

class GaussianTestCase(unittest.TestCase):
    """
    Test the gaussian model
    """
    def setUp(self):
        self.model=GaussianModel(dim = 50)
        self.work=raynest.raynest(self.model, verbose=2, nensemble=0, nlive=1000, maxmcmc=5000, nslice=6, nhamiltonian=0, resume=0)

    def test_run(self):
        self.work.run()
        # 2 sigma tolerance
        tolerance = 2.0*np.sqrt(self.work.NS.info/self.work.NS.nlive)
        self.assertTrue(np.abs(self.work.NS.logZ - self.model.analytic_log_Z)<tolerance, 'Incorrect evidence for normalised distribution: {0} instead of {1}'.format(self.work.NS.logZ ,self.model.analytic_log_Z))
        print("analytic logZ = {0}".format(self.model.analytic_log_Z))
def test_all():
    unittest.main(verbosity=2)

if __name__=='__main__':
#    unittest.main(verbosity=2)
    model=GaussianModel(dim = 50)
    work=raynest.raynest(model, verbose=2,
                       nnest=1, nensemble=3, nlive=1000, maxmcmc=5000, nslice=0, nhamiltonian=0, seed = 1,
                       resume=0, periodic_checkpoint_interval=600)
    work.run()
    print("analytic logZ = {0}".format(model.analytic_log_Z))
    print("estimated logZ = {0} +- {1}".format(work.logZ,work.logZ_error))
