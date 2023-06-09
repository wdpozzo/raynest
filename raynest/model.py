from abc import ABCMeta,abstractmethod,abstractproperty
from numpy import inf
import numpy as np
from .parameter import LivePoint

import logging
LOGGER = logging.getLogger('raynest.model')  # <--- for module-level logging


class Model(object):
    """
    Base class for user's model. User should subclass this
    and implement log_likelihood, names and bounds
    """
    __metaclass__ = ABCMeta
    names=[] # Names of parameters, e.g. ['p1','p2']
    bounds=[] # Bounds of prior as list of tuples, e.g. [(min1,max1), (min2,max2), ...]

    def in_bounds(self,param):
        """
        Checks whether param lies within the bounds

        -----------
        Parameters:
            param: :obj:`raynest.parameter.LivePoint`

        -----------
        Return:
            True: if all dimensions are within the bounds
            False: otherwise
        """
        return all(self.bounds[i][0] < param.values[i] < self.bounds[i][1] for i in range(param.dimension))

    def new_point(self, rng = None):
        """
        Create a new LivePoint, drawn from within bounds

        -----------
        Return:
            p: :obj:`raynest.parameter.LivePoint`
        """
        
        if rng is None:
            generator = np.random.uniform
        else:
            generator = rng.uniform
        logP = -inf
        
        while(logP==-inf):
            p = LivePoint(self.names,
                          d=np.array([generator(self.bounds[i][0],
                                                self.bounds[i][1])
                                       for i, _ in enumerate(self.names) ]
                                    )
                         )
            logP=self.log_prior(p)
        return p

    @abstractmethod
    def log_likelihood(self,param):
        """
        returns log likelihood of given parameter

        ------------
        Parameter:
            param: :obj:`raynest.parameter.LivePoint`
        """
        pass

    def log_prior(self,param):
        """
        Returns log of prior.
        Default is flat prior within bounds

        ----------
        Parameter:
            param: :obj:`raynest.parameter.LivePoint`

        ----------
        Return:
            0 if param is in bounds
            -np.inf otherwise
        """
        if self.in_bounds(param):
            return 0.0
        else: return -inf

    def potential(self,param):
        """
        returns the potential energy as minus the log prior
        ----------
        Parameter:
        param: :obj:`raynest.parameter.LivePoint`

        ----------
        Return:
            :obj: -`raynest.model.log_prior`
        """
        return -self.log_prior(param)
    
    @abstractmethod
    def force(self,param):
        """
        returns the force (-grad potential)
        Required for Hamiltonian sampling

        ----------
        Parameter:
        param: :obj:`raynest.parameter.LivePoint`
        """
        pass

    @abstractmethod
    def analytical_gradient(self,param):
        """
        returns the gradient of the likelihood (-grad potential)
        Required for Hamiltonian sampling

        ----------
        Parameter:
        param: :obj:`raynest.parameter.LivePoint`
        """
        pass

    def strsample(self,sample):
        """
        Return a string representation for the sample to be written
        to the output file. User may overload for additional output
        ----------
        Parameter:
            param: :obj:`raynest.parameter.LivePoint`
        ----------
        Return:
            line: :string:
        """
        line='\t'.join('{0:.20e}'.format(sample[n]) for n in sample.names)
        line+='{0:20e}'.format(sample.logL)
        return line

    def header(self):
        """
        Return a string with the output file header
        """
        return '\t'.join(self.names) + '\tlogL'

    def from_normalised(self, normalised_value):
        """
        Maps from [0,1]^Ndim to the full range of the parameters
        Inverse of to_normalised()
        ----------
        Parameter:
            normalised_vaue: array-like values in range (0,1)
        ----------
        Returns:
            point: :obj:`raynest.parameter.LivePoint`
        """
        d=np.array([self.bounds[i][0]
                 + normalised_value[i] * (self.bounds[i][1] - self.bounds[i][0])
                 for i, _ in enumerate(self.names)]
                 )
        return LivePoint(self.names, d=d)

    def to_normalised(self, point):
        """
        Maps the bounds of the parameters onto [0,1]
        ----------
        Parameter:
            point: :obj:`raynest.parameter.LivePoint`
        ----------
        Returns:
            normalised_value: :obj:`array.array`
                The values of the parameter mapped into the Ndim-cube
        """
        return np.array([(v-b[0])/(b[1]-b[0]) for v, b
                           in zip(point.values, self.bounds)])
