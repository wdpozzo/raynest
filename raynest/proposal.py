from __future__ import division
from functools import reduce
import numpy as np
from math import log,sqrt,fabs,exp
from abc import ABCMeta,abstractmethod
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
from .nest2pos import acl
from .parameter import LivePoint
from .NestedSampling import LivePoints
import ray
    
class Proposal(object):
    """
    Base abstract class for jump proposals
    """
    __metaclass__ = ABCMeta
    log_J = 0.0 # Jacobian of this jump proposal
    rng   = None # numpy Generator object
    
    def __init__(self, rng, *args, **kwargs):
        self.rng = rng
    
    @abstractmethod
    def get_sample(self,old):
        """
        Returns a new proposed sample given the old one.
        Must be implemented by user

        Parameters
        ----------
        old : :obj:`raynest.parameter.LivePoint`

        Returns
        ----------
        out: :obj:`raynest.parameter.LivePoint`
        """
        pass

class EnsembleProposal(Proposal):
    """
    Base class for ensemble proposals
    """
    ensemble=None
    
    def set_ensemble(self,ensemble):
        """
        Set the ensemble of points to use
        """
        self.ensemble = ensemble
        
class ProposalCycle(EnsembleProposal):
    """
    A proposal that cycles through a list of
    jumps.

    Initialisation arguments:

    proposals : A list of jump proposals
    weights   : Weights for each type of jump

    Optional arguments:
    cyclelength : length of the proposal cycle. Default: 100

    """
    idx=0 # index in the cycle
    N=0   # number of proposals in the cycle
    def __init__(self,proposals,weights,rng,cyclelength=100,*args,**kwargs):
        super(ProposalCycle,self).__init__(rng)
        assert(len(weights)==len(proposals))
        self.rng         = rng
        self.cyclelength = cyclelength
        self.weights     = weights
        self.proposals   = proposals
        self.set_cycle()

    def set_cycle(self):
        # The cycle is a list of indices for self.proposals
        self.cycle = self.rng.choice(self.proposals, size=self.cyclelength,
                                     p=self.weights, replace=True)
        self.N=len(self.cycle)

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights):
        self._weights = self.normalise_weights(weights)

    def normalise_weights(self, weights):
        norm = sum(weights)
        for i, _ in enumerate(weights):
            weights[i]=weights[i] / norm
        return weights

    def get_sample(self,old,**kwargs):
        # Call the current proposal and increment the index
        self.idx = (self.idx + 1) % self.N
        p = self.cycle[self.idx]
        new = p.get_sample(old,**kwargs)
        self.log_J = p.log_J
        return new

    def set_ensemble(self,ensemble):
        """
        Updates the ensemble statistics
        by calling it on each :obj:`EnsembleProposal`
        """
        for p in self.proposals:
            if isinstance(p,EnsembleProposal):
                p.set_ensemble(ensemble)
        del ensemble

    def add_proposal(self, proposal, weight):
        self.proposals = self.proposals + [proposal]
        self.weights = self.weights + [weight]
        self.set_cycle()

class EnsembleSlice(EnsembleProposal):
    """
    The Ensemble Slice proposal from Karamanis & Beutler
    https://arxiv.org/pdf/2002.06212v1.pdf
    """
    log_J      = 0.0 # Symmetric proposal
    mean       = None
    covariance = None
            
    def set_ensemble(self,ensemble):
        """
        Over-ride default set_ensemble so that the
        mean and covariance matrix are recomputed when it is updated
        """
        super(EnsembleSlice,self).set_ensemble(ensemble)
        self.mean, self.covariance = ensemble.get_mean_covariance()
        
class EnsembleSliceDifferential(EnsembleSlice):
    """
    The Ensemble Slice Differential move from Karamanis & Beutler
    https://arxiv.org/pdf/2002.06212v1.pdf
    """

    def get_direction(self, mu = 1.0):
        """
        Draws two random points and returns their direction
        """
        subset = self.ensemble.sample(2)
        direction = reduce(LivePoint.__sub__,subset)
        return direction * mu

class EnsembleSliceCorrelatedGaussian(EnsembleSlice):
    """
    The Ensemble Slice Correlated Gaussian move from Karamanis & Beutler
    https://arxiv.org/pdf/2002.06212v1.pdf
    """
    def get_direction(self, mu = 1.0):
        """
        Draws a random gaussian direction
        """
        direction = mu * self.rng.multivariate_normal(self.mean, self.covariance)
        return direction

class EnsembleSliceGaussian(EnsembleSlice):
    """
    The Ensemble Slice Gaussian move from Karamanis & Beutler
    https://arxiv.org/pdf/2002.06212v1.pdf
    """

    def get_direction(self, mu = 1.0):
        """
        Draw a random gaussian direction
        """
        direction  = self.rng.normal(0.0,1.0,size=len(self.mean))
        direction /= np.linalg.norm(direction)
        return direction * mu

class EnsembleSliceProposalCycle(ProposalCycle):
    def __init__(self, rng, model=None):
        """
        A proposal cycle that uses the slice sampler :obj:`EnsembleSlice`
        proposal.
        """
        weights = [1,1,1]
        proposals = [EnsembleSliceDifferential(rng),EnsembleSliceGaussian(rng),EnsembleSliceCorrelatedGaussian(rng)]
        super(EnsembleSliceProposalCycle,self).__init__(proposals, weights, rng)

    def get_direction(self, mu = 1.0, **kwargs):
        """
        Get a direction for the slice jump
        """
        self.idx = (self.idx + 1) % self.N
        p = self.cycle[self.idx]
        new = p.get_direction(mu = mu, **kwargs)
        self.log_J = p.log_J
        return new

class EnsembleWalk(EnsembleProposal):
    """
    The Ensemble "walk" move from Goodman & Weare
    http://dx.doi.org/10.2140/camcos.2010.5.65

    Draws a step by evolving along the
    direction of the center of mass of
    3 points in the ensemble.
    """
    log_J = 0.0 # Symmetric proposal
    Npoints = 3
    def get_sample(self,old):
        """
        Parameters
        ----------
        old : :obj:`raynest.parameter.LivePoint`

        Returns
        ----------
        out: :obj:`raynest.parameter.LivePoint`
        """
        subset = self.ensemble.sample(self.Npoints)
        center_of_mass = reduce(type(old).__add__,subset)/float(self.Npoints)
        out = old
        for x in subset:
            out += (x - center_of_mass)*self.rng.standard_normal()
        return out

class EnsembleModifiedWalk(EnsembleProposal):
    """
    The Ensemble "walk" move from Goodman & Weare
    http://dx.doi.org/10.2140/camcos.2010.5.65
    modified as in https://arxiv.org/pdf/2411.00276
    """
    log_J = 0.0 # Symmetric proposal
    Npoints = 3
    tj = -1.0
    tk = +1.0
    
    def lagrange_weights(self, x, x0, x1, x2):
        return (x-x1)/(x0-x1) * (x-x2)/(x0-x2)
        
    def get_sample(self,old):
        """
        Parameters
        ----------
        old : :obj:`raynest.parameter.LivePoint`

        Returns
        ----------
        out: :obj:`raynest.parameter.LivePoint`
        """
        subset_1 = self.ensemble.sample(self.Npoints)
        center_of_mass_1 = reduce(type(old).__add__,subset_1)/float(self.Npoints)
        subset_2 = self.ensemble.sample(self.Npoints)
        center_of_mass_2 = reduce(type(old).__add__,subset_2)/float(self.Npoints)
        
        ti   = self.rng.standard_normal()
        tNew = self.rng.standard_normal()
        
        wi = self.lagrange_weights(tNew,ti,self.tj,self.tk);
        wj = self.lagrange_weights(tNew,self.tj,self.tk,ti);
        wk = self.lagrange_weights(tNew,self.tk,ti,self.tj);
        
        out = old.copy()
        out.values = wi*old.values + wj*center_of_mass_1.values + wk*center_of_mass_2.values
        # Jacobian
        self.log_J = out.dimension * np.log(np.abs(wi))
        return out

class EnsembleStretch(EnsembleProposal):
    """
    The Ensemble "stretch" move from Goodman & Weare
    http://dx.doi.org/10.2140/camcos.2010.5.65
    """
    def get_sample(self,old):
        """
        Parameters
        ----------
        old : :obj:`raynest.parameter.LivePoint`

        Returns
        ----------
        out: :obj:`raynest.parameter.LivePoint`
        """
        scale = 2.0 # Will stretch factor in (1/scale,scale)
        # Pick a random point to move toward
        a = self.ensemble.sample(1)[0]
        # Pick the scale factor
        x = self.rng.uniform(-1,1)*np.log(scale)
        Z = exp(x)
        out = a + (old - a)*Z
        # Jacobian
        self.log_J = out.dimension * x
        return out

class EnsembleModifiedStretch(EnsembleProposal):
    """
    The Ensemble "stretch" move from Goodman & Weare
    http://dx.doi.org/10.2140/camcos.2010.5.65
    modified as in https://arxiv.org/pdf/2411.00276
    """
    def get_sample(self,old):
        """
        Parameters
        ----------
        old : :obj:`raynest.parameter.LivePoint`

        Returns
        ----------
        out: :obj:`raynest.parameter.LivePoint`
        """
        scale  = 2.0 # Will stretch factor in (1/scale,scale)
        # Pick a random point to move toward
        a, b   = self.ensemble.sample(2)
        c      = self.rng.uniform(-1,1)
        r_star = c*a.values+(1-c)*b.values
        # Pick the scale factor
        x = self.rng.uniform(-1,1)*np.log(scale)
        Z = exp(x)
        out = old.copy()
        out.values = Z*old.values + (1-Z)*r_star
        # Jacobian
        self.log_J = out.dimension * x
        return out
        
class DifferentialEvolution(EnsembleProposal):
    """
    Differential evolution move:
    Draws a step by taking the difference vector between two points in the
    ensemble and adding it to the current point.
    See e.g. Exercise 30.12, p.398 in MacKay's book
    http://www.inference.phy.cam.ac.uk/mackay/itila/

    We add a small perturbation around the exact step
    """
    log_J = 0.0 # Symmetric jump
    def get_sample(self,old):
        """
        Parameters
        ----------
        old : :obj:`raynest.parameter.LivePoint`

        Returns
        ----------
        out: :obj:`raynest.parameter.LivePoint`
        """
        a, b = self.ensemble.sample(2)
        sigma = 1e-4 # scatter around difference vector by this factor
        out = old + (b-a)*self.rng.normal(1.0,sigma)
        return out

class EnsembleEigenVector(EnsembleProposal):
    """
    A jump along a randomly-chosen eigenvector
    of the covariance matrix of the ensemble
    """
    log_J         = 0.0
    eigen_values  = None
    eigen_vectors = None
    covariance    = None
    ensemble      = None
    def set_ensemble(self, ensemble):
        """
        Over-ride default set_ensemble so that the
        eigenvectors are recomputed when it is updated
        """
        self.update_eigenvectors(ensemble)

    def update_eigenvectors(self, ensemble):
        """
        Recompute the eigenvectors and eigevalues
        of the covariance matrix of the ensemble
        """
        self.eigen_values, self.eigen_vectors = ensemble.get_eigen_quantities()

    def get_sample(self,old):
        """
        Propose a jump along a random eigenvector
        Parameters
        ----------
        old : :obj:`raynest.parameter.LivePoint`

        Returns
        ----------
        out: :obj:`raynest.parameter.LivePoint`
        """
        out = old.copy()
        # pick a random eigenvector
        i = self.rng.integers(low=0, high=old.dimension)
        jumpsize = sqrt(fabs(self.eigen_values[i]))*self.rng.standard_normal()
        out.values += jumpsize*self.eigen_vectors[:,i]
        return out

class EnsembleQuadratic(EnsembleProposal):

    tj = -1.0
    tk = +1.0
    
    def lagrange_weights(self, x, x0, x1, x2):
        return (x-x1)/(x0-x1) * (x-x2)/(x0-x2)

    def get_sample(self,old):
        """
        Propose a jump using the quadratic move in Militzer (2023a)
        ----------
        old : :obj:`raynest.parameter.LivePoint`

        Returns
        ----------
        out: :obj:`raynest.parameter.LivePoint`
        """
        a, b = self.ensemble.sample(2)

        ti   = self.rng.standard_normal()
        tNew = self.rng.standard_normal()
        
        wi = self.lagrange_weights(tNew,ti,self.tj,self.tk);
        wj = self.lagrange_weights(tNew,self.tj,self.tk,ti);
        wk = self.lagrange_weights(tNew,self.tk,ti,self.tj);
        out = old.copy()
        out.values = wi*old.values + wj*a.values + wk*b.values
        # Jacobian
        self.log_J = out.dimension * np.log(np.abs(wi))
        return out

class DefaultProposalCycle(ProposalCycle):
    """
    A default proposal cycle that uses the
    :obj:`raynest.proposal.EnsembleWalk`, :obj:`raynest.proposal.EnsembleStretch`,
    :obj:`raynest.proposal.DifferentialEvolution`, :obj:`raynest.proposal.EnsembleEigenVector`
    ensemble proposals.
    """
    def __init__(self, rng, *args, **kwargs):

        proposals = [EnsembleWalk(rng),
                     EnsembleModifiedWalk(rng),
                     EnsembleStretch(rng),
                     EnsembleModifiedStretch(rng),
                     EnsembleQuadratic(rng),
                     DifferentialEvolution(rng),
                     EnsembleEigenVector(rng)]
        weights = [3,
                   1,
                   3,
                   1,
                   5,
                   3,
                   10]

        super(DefaultProposalCycle,self).__init__(proposals, weights, rng)

class HamiltonianProposalCycle(ProposalCycle):
    def __init__(self, rng, model=None, density=None):
        """
        A proposal cycle that uses the hamiltonian :obj:`ConstrainedLeapFrog`
        proposal.
        Requires a :obj:`raynest.Model` to be passed for access to the user-defined
        :obj:`raynest.Model.force` (the gradient of :obj:`raynest.Model.potential`) and
        :obj:`raynest.Model.log_likelihood` to define the reflective
        """
        weights = [1]
        proposals = [ConstrainedLeapFrog(rng, model=model, density=density)]
        super(HamiltonianProposalCycle,self).__init__(proposals, weights, rng)

class HamiltonianProposal(EnsembleProposal):
    """
    Base class for hamiltonian proposals
    """
    covariance           = None
    mass_matrix          = None
    inverse_mass_matrix  = None
    momenta_distribution = None

    def __init__(self, rng, model=None, density=None):
        """
        Initialises the class with the kinetic
        energy and the :obj:`raynest.Model.potential`.
        """
        super(HamiltonianProposal, self).__init__(rng)
        self.rng                    = rng
        self.T                      = self.kinetic_energy
        self.model                  = model
        self.V                      = self.model.potential
        self.prior_bounds           = self.model.bounds
        self.dimension              = len(self.prior_bounds)
        self.dt                     = 0.3
        self.leaps                  = 100
        self.maxleaps               = 1000
        self.DEBUG                  = 0
        self.trajectories           = []
        self.covariance             = np.identity(self.dimension)
        
        self.set_mass_parameters()
        self.set_momenta_distribution()
    
    def set_mass_parameters(self):
        self.mass_matrix         = self.covariance
        self.inverse_mass_matrix = np.linalg.inv(self.mass_matrix)
        self.inverse_mass        = np.atleast_1d(np.squeeze(np.diag(self.inverse_mass_matrix)))
        _, self.logdeterminant   = np.linalg.slogdet(self.mass_matrix)
        
    def set_ensemble(self, density_covariance):
        """
        FIXME:
        """
        self.density = density_covariance[0]
#        self.covariance = density_covariance[1]
#        self.set_mass_parameters()

    def unit_normal(self, q):
        v = self.density._log_gradient(q.values)
        return v/np.linalg.norm(v)

    def force(self, q):
        """
        return the gradient of the potential function as numpy ndarray
        Parameters
        ----------
        q : :obj:`raynest.parameter.LivePoint`
            position

        Returns
        ----------
        dV: :obj:`numpy.ndarray` gradient evaluated at q
        """
        return -self.density._log_gradient(q.values)

    def set_momenta_distribution(self):
        """
        update the momenta distribution using the
        mass matrix (precision matrix of the ensemble).
        """
        self.momenta_distribution = multivariate_normal(cov=self.mass_matrix)
        
    def kinetic_energy(self,p):
        """
        kinetic energy part for the Hamiltonian.
        Parameters
        ----------
        p : :obj:`numpy.ndarray`
            momentum

        Returns
        ----------
        T: :float: kinetic energy
        """
        return 0.5 * np.dot(p,np.dot(self.inverse_mass_matrix,p))-self.logdeterminant

    def hamiltonian(self, p, q):
        """
        Hamiltonian.
        Parameters
        ----------
        p : :obj:`numpy.ndarray`
            momentum
        q : :obj:`raynest.parameter.LivePoint`
            position
        Returns
        ----------
        H: :float: hamiltonian
        """
        return self.T(p) + self.V(q)

class LeapFrog(HamiltonianProposal):
    """
    Leap frog integrator proposal for an unconstrained
    Hamiltonian Monte Carlo step
    """
    def __init__(self, *args, model=None, density=None):
        """
        Parameters
        ----------
        model : :obj:`raynest.Model`
        """
        super(LeapFrog, self).__init__(model=model, density=density)
        self.prior_bounds   = model.bounds

    def get_sample(self, q0, *args):
        """
        Propose a new sample, starting at q0

        Parameters
        ----------
        q0 : :obj:`raynest.parameter.LivePoint`
            position

        Returns
        ----------
        q: :obj:`raynest.parameter.LivePoint`
            position
        """
        # compute local equilibrium distribution
#        self.set_lte(q0)
        # generate a canonical momentum
        p0 = np.atleast_1d(self.momenta_distribution.rvs())
        initial_energy = self.hamiltonian(p0,q0)
        # evolve along the trajectory
        q, p, r = self.evolve_trajectory(p0, q0, *args)
        # minus sign from the definition of the potential
        final_energy   = self.hamiltonian(p, q)
        if r == 1:
            self.log_J = -np.inf
        else:
            self.log_J = min(0.0, initial_energy-final_energy)
        return q

    def evolve_trajectory(self, p0, q0, *args):
        """
        Hamiltonian leap frog trajectory subject to the
        hard boundary defined by the parameters prior bounds.
        https://arxiv.org/pdf/1206.1901.pdf

        Parameters
        ----------
        p0 : :obj:`numpy.ndarray`
            momentum
        q0 : :obj:`raynest.parameter.LivePoint`
            position

        Returns
        ----------
        p: :obj:`numpy.ndarray` updated momentum vector
        q: :obj:`raynest.parameter.LivePoint`
            position
        """
        # Updating the momentum a half-step
        p = p0 - 0.5 * self.dt * self.gradient(q0)
        q = q0.copy()
        
        for i in range(self.leaps):

            # do a step
            for j,k in enumerate(q.names):
                u,l = self.prior_bounds[j][1], self.prior_bounds[j][0]
                q[k] += self.dt * p[j] * self.inverse_mass[j]
                # check and reflect against the bounds
                # of the allowed parameter range
                while q[k] <= l or q[k] >= u:
                    if q[k] > u:
                        q[k] = u - (q[k] - u)
                        p[j] *= -1
                    if q[k] < l:
                        q[k] = l + (l - q[k])
                        p[j] *= -1

            dV = self.gradient(q)
            # take a full momentum step
            p += - self.dt * dV
        # Do a final update of the momentum for a half step
        p += - 0.5 * self.dt * dV

        return q, -p, 0

class ConstrainedLeapFrog(HamiltonianProposal):
    """
    Leap frog integrator proposal for a costrained
    (logLmin defines a reflective boundary)
    Hamiltonian Monte Carlo step.
    """
    def __init__(self, *args, model=None, density=None):
        """
        Parameters
        ----------
        model : :obj:`raynest.Model`
        """
        super(ConstrainedLeapFrog, self).__init__(*args, model=model, density=density)
        self.log_likelihood = model.log_likelihood
        self.density        = density
        
    def get_sample(self, q0, logLmin=-np.inf):
        """
        Generate new sample with constrained HMC, starting at q0.

        Parameters
        ----------
        q0 : :obj:`raynest.parameter.LivePoint`
            position

        logLmin: hard likelihood boundary

        Returns
        ----------
        q: :obj:`raynest.parameter.LivePoint`
            position
        """
        p0 = np.atleast_1d(self.momenta_distribution.rvs())
        initial_energy = self.hamiltonian(p0,q0)
        # evolve along the trajectory
        q, p, r = self.evolve_trajectory(p0, q0, logLmin)
        # minus sign from the definition of the potential
        final_energy   = self.hamiltonian(p, q)
        if r == 1:
            self.log_J = -np.inf
        else:
            self.log_J = min(0.0, initial_energy-final_energy)
        return q

    def counter(self):
        n = 0
        while True:
            yield n
            n += 1

    def evolve_trajectory_one_step_position(self, p, q):
        """
        One leap frog step in position

        Parameters
        ----------
        p0 : :obj:`numpy.ndarray`
            momentum
        q0 : :obj:`raynest.parameter.LivePoint`
            position
        Returns
        ----------
        p: :obj:`numpy.ndarray` updated momentum vector
        q: :obj:`raynest.parameter.LivePoint` position
        """
        for j,k in enumerate(q.names):
            u, l  = self.prior_bounds[j][1], self.prior_bounds[j][0]
            q[k]  += self.dt * p[j] * self.inverse_mass[j]
            # check and reflect against the bounds
            # of the allowed parameter range
            while q[k] < l or q[k] > u:
                if q[k] > u:
                    q[k] = u - (q[k] - u)
                    p[j] *= -1
                if q[k] < l:
                    q[k] = l + (l - q[k])
                    p[j] *= -1
        return p, q

    def evolve_trajectory_one_step_momentum(self, p, q, logLmin, half = False):
        """
        One leap frog step in momentum

        Parameters
        ----------
        p0 : :obj:`numpy.ndarray`
            momentum
        q0 : :obj:`raynest.parameter.LivePoint`
            position
        logLmin: :obj:`numpy.float64`
            loglikelihood constraint
        Returns
        ----------
        p: :obj:`numpy.ndarray` updated momentum vector
        q: :obj:`raynest.parameter.LivePoint` position
        """
        reflected = 0
        dV        = self.force(q)
        if half is True:
            p += - 0.5 * self.dt * dV
            return p, q, reflected
        else:
            c = self.check_constraint(q, logLmin)
            if c > 0:
                p += - self.dt * dV
            else:
                normal = self.unit_normal(q)
                p += - 2.0*np.dot(p,normal)*normal
                reflected = 1
        return p, q, reflected

    def check_constraint(self, q, logLmin):
        """
        Check the likelihood

        Parameters
        ----------
        q0 : :obj:`raynest.parameter.LivePoint`
        position
        logLmin: :obj:`numpy.float64`
        loglikelihood constraint
        Returns
        ----------
        c: :obj:`numpy.float64` value of the constraint
        """
        q.logP  = -self.V(q)
        q.logL  = self.log_likelihood(q)
        return q.logL - logLmin

    def evolve_trajectory(self, p0, q0, logLmin):
        """
        Evolve point according to Hamiltonian method in
        https://arxiv.org/pdf/1005.0157.pdf

        Parameters
        ----------
        p0 : :obj:`numpy.ndarray`
            momentum
        q0 : :obj:`raynest.parameter.LivePoint`
            position

        Returns
        ----------
        p: :obj:`numpy.ndarray` updated momentum vector
        q: :obj:`raynest.parameter.LivePoint` position
        """
        trajectory = [(q0,p0,0)]
        # evolve forward in time
        i = 0
        p, q, reflected = self.evolve_trajectory_one_step_momentum(p0.copy(), q0.copy(), logLmin, half = True)
        while (i < self.leaps):
            p, q            = self.evolve_trajectory_one_step_position(p, q)
            p, q, reflected = self.evolve_trajectory_one_step_momentum(p, q, logLmin, half = False)
            trajectory.append((q.copy(),p.copy(),reflected))
            i += 1

        p, q, reflected     = self.evolve_trajectory_one_step_momentum(p, q, logLmin, half = True)
        if self.DEBUG: self.save_trajectory(trajectory, logLmin)

#        # evolve backward in time
#        i = 0
#        p, q, reflected = self.evolve_trajectory_one_step_momentum(-p0.copy(), q0.copy(), logLmin, half = True)
#        while (i < self.leaps//2):
#            p, q            = self.evolve_trajectory_one_step_position(p, q)
#            p, q, reflected = self.evolve_trajectory_one_step_momentum(p, q, logLmin, half = False)
#            trajectory.append((q.copy(),p.copy(),reflected))
#            i += 1
#
#        p, q, reflected     = self.evolve_trajectory_one_step_momentum(p, q, logLmin, half = True)
#
#        if self.DEBUG: self.save_trajectory(trajectory, logLmin)
#        q, p, reflected = self.sample_trajectory(trajectory)
#
#        self.trajectories.append(trajectory)
        return q, -p, reflected

    def sample_trajectory(self, trajectory):
        """

        """
        logw = np.array([-self.hamiltonian(p,q) for q,p,_ in trajectory[1:-1]])
        norm = logsumexp(logw)
        idx  = self.rng.choice(range(1,len(trajectory)-1), p = np.exp(logw  - norm))
        return trajectory[idx]

    def save_trajectory(self, trajectory, logLmin, filename = None):
        """
        save trajectory for diagnostic purposes
        """
        if filename is None:
            filename = 'trajectory_'+str(next(self.c))+'.txt'
        f = open(filename,'w')
        names = trajectory[0][0].names

        for n in names:
            f.write(n+'\t'+'p_'+n+'\t')
        f.write('logPrior\tlogL\tlogLmin\n')

        for j,step in enumerate(trajectory):
            q = step[0]
            p = step[1]
            for j,n in enumerate(names):
                f.write(repr(q[n])+'\t'+repr(p[j])+'\t')
            f.write(repr(q.logP)+'\t'+repr(q.logL)+'\t'+repr(logLmin)+'\n')
        f.close()
        if self.c == 3: exit()

class NoUTurn(ConstrainedLeapFrog):
    def build_tree(theta, r, grad, v, j, epsilon, f, joint0):
        """The main recursion."""
        if (j == 0):
            # Base case: Take a single leapfrog step in the direction v.
            thetaprime, rprime, gradprime, logpprime = leapfrog(theta, r, grad, v * epsilon, f)
            jointprime = logpprime - 0.5 * np.dot(rprime, rprime.T)
            # Is the simulation wildly inaccurate?
            sprime = jointprime - joint0 > -1000
            # Set the return values---minus=plus for all things here, since the
            # "tree" is of depth 0.
            thetaminus = thetaprime[:]
            thetaplus = thetaprime[:]
            rminus = rprime[:]
            rplus = rprime[:]
            gradminus = gradprime[:]
            gradplus = gradprime[:]
            logptree = jointprime - joint0
            #logptree = logpprime
            # Compute the acceptance probability.
            alphaprime = min(1., np.exp(jointprime - joint0))
            #alphaprime = min(1., np.exp(logpprime - 0.5 * np.dot(rprime, rprime.T) - joint0))
            nalphaprime = 1
        else:
            # Recursion: Implicitly build the height j-1 left and right subtrees.
            thetaminus, rminus, gradminus, thetaplus, rplus, gradplus, thetaprime, gradprime, logpprime, sprime, alphaprime, nalphaprime, logptree = build_tree(theta, r, grad, v, j - 1, epsilon, f, joint0)
            # No need to keep going if the stopping criteria were met in the first subtree.
            if sprime:
                if v == -1:
                    thetaminus, rminus, gradminus, _, _, _, thetaprime2, gradprime2, logpprime2, sprime2, alphaprime2, nalphaprime2, logptree2 = build_tree(thetaminus, rminus, gradminus, v, j - 1, epsilon, f, joint0)
                else:
                    _, _, _, thetaplus, rplus, gradplus, thetaprime2, gradprime2, logpprime2, sprime2, alphaprime2, nalphaprime2, logptree2 = build_tree(thetaplus, rplus, gradplus, v, j - 1, epsilon, f, joint0)
                # Conpute total probability of this trajectory
                logptot = np.logaddexp(logptree, logptree2)
                # Choose which subtree to propagate a sample up from.
                if np.log(np.random.uniform()) < logptree2 - logptot:
                    thetaprime = thetaprime2[:]
                    gradprime = gradprime2[:]
                    logpprime = logpprime2
                logptree = logptot
                # Update the stopping criterion.
                sprime = sprime and sprime2 and stop_criterion(thetaminus, thetaplus, rminus, rplus)
                # Update the acceptance probability statistics.
                alphaprime = alphaprime + alphaprime2
                nalphaprime = nalphaprime + nalphaprime2

        return thetaminus, rminus, gradminus, thetaplus, rplus, gradplus, thetaprime, gradprime, logpprime, sprime, alphaprime, nalphaprime, logptree

def tree_sample(theta, logp, r0, grad, epsilon, f, joint, maxheight=np.inf):
    pass
