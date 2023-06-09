import unittest
import numpy as np
import raynest.model

class GaussianModel(raynest.model.Model):
    """
    A simple 2 dimensional gaussian
    """
    def __init__(self):
        pass
    names=['x','y']
    bounds=[[-10,10],[-10,10]]
    analytic_log_Z=0.0 - np.log(bounds[0][1]-bounds[0][0]) - np.log(bounds[1][1]-bounds[1][0])

    def log_likelihood(self,p):
        return -0.5*(p['x']**2 + p['y']**2) - np.log(2.0*np.pi)

    def log_prior(self,p):
        return super(GaussianModel,self).log_prior(p)
    
    def force(self, p):
        f = np.zeros(1, dtype = {'names':p.names, 'formats':['f8' for _ in p.names]})
        return f

class GaussianTestCase(unittest.TestCase):
    """
    Test the gaussian model
    """
    def setUp(self):
        gaussmodel = GaussianModel()
        self.work=raynest.raynest(gaussmodel,verbose=2,nensemble=2,nlive=1000,maxmcmc=500,nslice=0,nhamiltonian=0)
        print('Sampling 2D gaussian with analytic evidence {0}'.format(gaussmodel.analytic_log_Z))

    def test_run(self):
        self.work.run()
        # 2 sigma tolerance
        logZ = self.work.logZ
        H    = self.work.information
        tolerance = 2.0*self.work.logZ_error
        self.assertTrue(np.abs(logZ - GaussianModel.analytic_log_Z)<tolerance, 'Incorrect evidence for normalised distribution: {0} +/- {2} instead of {1}'.format(logZ ,GaussianModel.analytic_log_Z, tolerance))

def test_all():
    unittest.main(verbosity=2)

if __name__=='__main__':
    unittest.main(verbosity=2)
 
