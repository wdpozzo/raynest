#!/bin/usr/env python

import numpy as np
import raynest.model

class RosenbrockModel(raynest.model.Model):
    """
    The n-dimensional Rosenbrock function

    See: https://arxiv.org/pdf/1903.09556.pdf
    """
    def __init__(self, ndims=2):
        self.ndims = ndims
        self.names = ['x' + str(i) for i in range(ndims)]
        self.bounds = [[-5, 5] for i in range(ndims)]


    def log_likelihood(self, x):
        x = np.array(x.values).reshape(-1, self.ndims)
        return  -(np.sum(100. * (x[:,1:] - x[:,:-1] ** 2.) ** 2. + (1. - x[:,:-1]) ** 2., axis=1))

    def force(self, x):
        f = np.zeros(1, dtype = {'names':x.names, 'formats':['f8' for _ in x.names]})
        return f

if __name__=='__main__':
    model=RosenbrockModel(ndims = 2)
    work=raynest.raynest(model,
                         verbose=2,
                         nnest=2,
                         nensemble=2,
                         nlive=100,
                         maxmcmc=5000,
                         nslice=0,
                         nhamiltonian=0,
                         seed = 1,
                         resume=0,
                         output='rosenbrock',
                         periodic_checkpoint_interval=600)
    work.run(corner = True)

