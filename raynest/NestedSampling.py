from __future__ import division, print_function
from asyncio import tasks
import sys
import os
import pickle
import time
import logging
import bisect
import numpy as np
from numpy import logaddexp
from numpy import inf
from scipy.stats import kstest
from math import isnan
from . import nest2pos
from .nest2pos import logsubexp
from operator import attrgetter
from .raynest import CheckPoint
from .utils import auto_garbage_collect
import copy

import ray
from tqdm import tqdm
logger = logging.getLogger('raynest.NestedSampling')

class _NSintegralState(object):
    """
    Stores the state of the nested sampling integrator
    """
    def __init__(self, nlive):
        self.nlive = nlive
        self.reset()
        loggername = 'raynest.NestedSampling._NSintegralState'
        self.logger = logging.getLogger(loggername)

    def reset(self):
        """
        Reset the sampler to its initial state at logZ = -infinity
        """
        self.iteration = 0
        self.logZ = -inf
        self.oldZ = -inf
        self.logw = 0
        self.info = 0
        # Start with a dummy sample enclosing the whole prior
        self.logLs = [-inf]  # Likelihoods sampled
        self.log_vols = [0.0]  # Volumes enclosed by contours

    def increment(self, logL, nlive=None, nreplace=1):
        """
        Increment the state of the evidence integrator
        Simply uses rectangle rule for initial estimate
        see: https://www.cell.com/biophysj/pdf/S0006-3495(12)00055-0.pdf
        for parallel implementation
        """
        if(logL<=self.logLs[-1]):
            self.logger.warning('NS integrator received non-monotonic logL. {0:.5f} -> {1:.5f}'.format(self.logLs[-1], logL))
        if nlive is None:
            nlive = self.nlive
        oldZ = self.logZ
        logt=-nreplace/nlive
        Wt = self.logw + logL + logsubexp(0,logt)
        self.logZ = logaddexp(self.logZ,Wt)
        # Update information estimate
        if np.isfinite(oldZ) and np.isfinite(self.logZ) and np.isfinite(logL):
            self.info = np.exp(Wt - self.logZ)*logL + np.exp(oldZ - self.logZ)*(self.info + oldZ) - self.logZ
            if isnan(self.info):
                self.info = 0

        # Update history
        self.logw += logt
        self.iteration += 1
        self.logLs.append(logL)
        self.log_vols.append(self.logw)

    def finalise(self):
        """
        Compute the final evidence with more accurate integrator
        Call at end of sampling run to refine estimate
        """
        from scipy import integrate
        # Trapezoidal rule
        self.logZ = nest2pos.log_integrate_log_trap(np.array(self.logLs), np.array(self.log_vols))
        return self.logZ

    def plot(self,filename):
        """
        Plot the logX vs logL
        """
        import matplotlib as mpl
        mpl.use('Agg')
        from matplotlib import pyplot as plt
        plt.figure()
        plt.plot(self.log_vols,self.logLs)
        plt.title('{0} iterations. logZ={1:.2f} H={2:.2f} bits'.format(self.iteration,self.logZ,self.info*np.log2(np.e)))
        plt.grid(which='both')
        plt.xlabel('log prior_volume')
        plt.ylabel('log likelihood')
        plt.xlim([self.log_vols[-1],self.log_vols[0]])
        plt.savefig(filename)
        self.logger.info('Saved nested sampling plot as {0}'.format(filename))

    def __getstate__(self):
        """Remove the unpicklable entries."""
        state = self.__dict__.copy()
        del state['logger']
        return state

    def __setstate__(self, state):
        if 'logger' not in state:
            loggername = "raynest.NestedSampling._NSintegralState"
            state['logger'] = logging.getLogger(loggername)
        self.__dict__ = state

class NestedSampler(object):
    """
    Nested Sampler class.
    Initialisation arguments:

    model: :obj:`raynest.Model` user defined model

    periodic_checkpoint_interval: float
        time interval in seconds between periodic checkpoints

    nthreads: int
        number of parallel sampling threads

    nlive: int
        number of live points to be used for the integration
        Default: 1024

    output: string
        folder where the output will be stored
        Default: None

    verbose: int
        0: Nothing
        1: display information on screen
        2: (1) + diagnostic plots
        Default: 1

    seed: int
        seed for the initialisation of the pseudorandom chain
        Default: 1234

    prior_sampling: boolean
        produce Nlive samples from the prior.
        Default: False

    stopping: float
        Stop when remaining samples wouldn't change logZ estimate by this much.
        Deafult: 0.1

    n_periodic_checkpoint: int
        **deprecated**
        This parameter should not be used, it should be set by the manager instead.
        checkpoint the sampler every n_periodic_checkpoint iterations
        Default: None (disabled)

    resume_file: string
        file where the checkpoints will be stored
        Default: None

    state: dict
        dictionary holding the checkpointed state
        Default: None
    """

    def __init__(self,
                 model,
                 periodic_checkpoint_interval = np.inf,
                 nthreads       = None,
                 nlive          = 1024,
                 output         = None,
                 verbose        = 1,
                 rng            = None,
                 prior_sampling = False,
                 stopping       = 0.1,
                 n_periodic_checkpoint = None,
                 position       = 0,
                 resume_file    = None,
                 state          = None
                 ):
        """
        Initialise all necessary arguments and
        variables for the algorithm
        """
        loggername          = 'raynest.NestedSampling.NestedSampler'
        self.logger         = logging.getLogger(loggername)
        self.logger.addHandler(logging.StreamHandler())
        self.model          = model
        self.dimension      = len(model.names)

        if state is None:
            if rng == None:
                self.rng = np.random.default_rng()
            else:
                self.rng            = rng
            self.position       = position
            self.periodic_checkpoint_interval = periodic_checkpoint_interval
            self.nthreads       = nthreads
            self.prior_sampling = prior_sampling
            self.verbose        = verbose
            self.acceptance     = 1.0
            self.accepted       = 0
            self.rejected       = 1
            self.nlive          = nlive
            self.live_points    = None
            self.last_checkpoint_time = time.time()
            self.tolerance      = stopping
            self.condition      = np.inf
            self.worst          = 0
            self.logLmin        = -np.inf
            self.logLmax        = -np.inf
            self.iteration      = 0
            self.prior_samples  = []
            self.nested_samples = []
            self.logLs          = []
            self.logPs          = []
            self.rolling_p      = []
            self.insertion_indices = []
            self.logZ           = None
            self.resume_file    = resume_file
            sys.stdout.flush()
            self.output_folder  = output
            self.output_file,self.evidence_file = self.setup_output(output)
            header              = open(os.path.join(output,'header.txt'),'w')
            header.write('\t'.join(self.model.names))
            header.write('\tlogL\n')
            header.close()
            self.integral_state    = _NSintegralState(self.nlive)
            self.initialise_live_points()
            self.initialised    = False
        else:
            for k,v in state.items():
                setattr(self,k,v)
            # silly workaround for extra live points if caught in the middle of
            # updating them
            if self.nlive != len(self.live_points._list):
                self.live_points._list.pop(-1)

    def initialise_live_points(self):

        self.live_points = np.zeros((self.nlive, self.dimension))
        self.logP = np.zeros(self.nlive)
        self.logL = np.zeros(self.nlive)
        
        with tqdm(total=self.nlive, disable = not self.verbose, desc='raynest: populate samplers', position=self.position) as pbar:
            for i in range(self.nlive):

                p, logP = self.model.new_point(rng = self.rng)
                logL = self.model.log_likelihood(p)
                self.live_points[i] = p
                self.logP[i] = logP
                self.logL[i] = logL
                pbar.update()
        #self.live_points = LivePoints(self.live_points, self.rng, n_replace=self.nthreads, verbose=self.verbose)

    def setup_output(self, output):
        """
        Set up the output folder

        -----------
        Parameters:
        output: string
            folder where the results will be stored
        -----------
        Returns:
            output_file, evidence_file, resume_file: tuple
                output_file:   file where the nested samples will be written
                evidence_file: file where the evidence will be written
                resume_file:   file used for checkpointing the algorithm
        """
        chain_filename = "chain_"+str(self.nlive)+"_"+str(self.position)+".txt"
        output_file   = os.path.join(output,chain_filename)
        evidence_file = os.path.join(output,chain_filename+"_evidence_"+str(self.position)+".txt")

        return output_file, evidence_file


    def write_chain_to_file(self):
        """
        Outputs a `raynest.parameter.LivePoint` to the
        output_file
        """
        with open(self.output_file,"w") as f:
            f.write('{0:s}\n'.format(self.model.header().rstrip()))
            for ns in self.nested_samples:
                line='\t'.join('{0:.20e}'.format(ns[n][0]) for n in self.output_names)
                f.write('{0:s}\n'.format(line.rstrip()))

    def write_evidence_to_file(self):
        """
        Write the evidence logZ and maximum likelihood to the evidence_file
        """
        with open(self.evidence_file,"w") as f:

            f.write('#logZ\tlogLmax\tH\n')
            f.write('{0:.5f} {1:.5f} {2:.2f}\n'.format(self.integral_state.logZ,
                                                       self.logLmax,
                                                       self.integral_state.info))

    def consume_sample(self, pool):
        """
        requests the workers to update the live points
        and updates the evidence logZ
        """

        lp_ref = ray.put(self.live_points)
        all_workers = []
        while True:
            worker = pool.pop_idle()
            if worker is None:
                break
            worker.set_ensemble.remote(lp_ref)
            all_workers.append(worker)

        nreplace = len(all_workers)

        # Get the worst live points
        # assume that they are already sorted according to logL
        self.worst   = self.live_points[:nreplace].copy()
        self.integral_state.increment(self.logL[nreplace-1], nreplace=nreplace)

        for i,w in enumerate(self.worst):
            self.append_nested_sample(w, self.logP[i], self.logL[i])
        
        # get the necessary statistics from the LivePoint actor
        self.logLmin, self.logLmax, logZ, info = self.logL[nreplace-1], self.logL[-1], self.integral_state.logZ, self.integral_state.info
        
        if self.verbose:
            logLtmp = self.logL[:nreplace]

        self.condition = logaddexp(logZ,self.logLmax - self.iteration/(float(self.nlive))) - logZ
        
        idx = self.rng.integers(low=nreplace, high=self.nlive, size=self.nthreads)
        starting_points = self.live_points[idx].copy()
        tasks = {}
        for i, w in enumerate(all_workers):
            ref = w.produce_sample.remote(starting_points[i], self.logLmin)
            tasks[ref] = i

        while tasks:
            ready, _ = ray.wait(list(tasks.keys()), num_returns=1)
            ref = ready[0]

            worker_index = tasks.pop(ref)
            result = ray.get(ref)

            acceptance, sub_acceptance, self.jumps, proposed, proposed_logP, proposed_logL = result

            if proposed_logL > self.logLmin:
                # replace worst point with new one
                insertion_index = replace_and_reinsert(self.live_points, self.logP, self.logL, worker_index, proposed, proposed_logP, proposed_logL)
                self.insertion_indices.append(insertion_index)
                self.accepted  += 1
                self.iteration += 1

                if self.verbose > 1:
                    self.logger.info("{0:d}: n:{1:4d} NS_acc:{2:.3f} S{3:02d}_acc:{4:.3f} sub_acc:{5:.3f} H: {6:.2f} logL {7:.5f} --> {8:.5f} dZ: {9:.3f} logZ: {10:.3f} logLmax: {11:.2f}"\
                        .format(self.iteration, self.jumps, self.acceptance, worker_index, acceptance, sub_acceptance, info,\
                        logLtmp[worker_index], proposed_logL, self.condition, logZ, self.logLmax))

            else:
                self.rejected += 1
                ref = all_workers[worker_index].produce_sample.remote(starting_points[worker_index], self.logLmin)
                tasks[ref] = worker_index

            self.acceptance = float(self.accepted)/float(self.accepted + self.rejected)

        for w in all_workers:
            pool.push(w)
        
    def reset(self, pool):
        """
        Initialise the pool of `raynest.parameter.LivePoint` by
        sampling them from the `raynest.model.log_prior` distribution
        """
        # set up  the ensemble statistics
        lp_ref = ray.put(self.live_points)
        for s in pool.map(lambda a, v: a.set_ensemble.remote(lp_ref), range(self.nthreads)):
            pass
        
        # send all live points to the samplers for start
        for i in range(self.nlive):
            pool.submit(lambda a, v: a.produce_sample.remote(v, -np.inf), self.live_points[i])

        i = 0
        self.output_names = self.model.names+['logPrior','logL']
        self.prior_samples = np.zeros(self.nlive, dtype = {'names':self.output_names, 'formats':['f8' for _ in self.output_names]})

        with tqdm(total=self.nlive, disable= not self.verbose, desc='raynest: sampling prior', position=self.position) as pbar:
            while pool.has_next():
                acceptance,sub_acceptance,self.jumps,x,logP,logL = pool.get_next()
                if np.isnan(logL):
                    self.logger.warning("Likelihood function returned NaN for params "+str(x))
                    self.logger.warning("You may want to check your likelihood function")
                if np.isfinite(logP) and np.isfinite(logL):
                    self.logP[i] = logP
                    self.logL[i] = logL
                    self.live_points[i] = x
                    for j, n in enumerate(self.model.names):
                        self.prior_samples[n][i] = x[j]
                    self.prior_samples['logL'][i] = logL
                    self.prior_samples['logPrior'][i] = logP
       
                    i += 1
                    pbar.update()
                else:
                    pool.submit(lambda a, v: a.produce_sample.remote(v, -np.inf), self.live_points[i])

        idx = np.argsort(self.logL)
        self.live_points = self.live_points[idx]
        self.logP = self.logP[idx]
        self.logL = self.logL[idx]

        if self.verbose:
            sys.stderr.write("\n")
            sys.stderr.flush()
        self.initialised=True

    def nested_sampling_loop(self, pool):
        """
        main nested sampling loop
        """
        if not self.initialised:
            self.reset(pool)
        else:
            self.logger.info("Nested Sampling process {0!s}, restoring samplers".format(os.getpid()))
            lp_ref = ray.put(self.live_points)
            for s in pool.map(lambda a, v: a.set_ensemble.remote(lp_ref), range(self.nthreads)):
                pass

        if self.prior_sampling:
            self.logZ = self.finalise()
            self.write_chain_to_file()
            self.write_evidence_to_file()
            self.logLmin = np.inf
            self.logLmax = np.inf
            self.logger.info("Nested Sampling process {0!s}, exiting".format(os.getpid()))
            return 0

        try:

            while self.condition > self.tolerance:
                self.consume_sample(pool)

                # update periodic checkpoint
                if time.time() - self.last_checkpoint_time > self.periodic_checkpoint_interval:
                    self.save()
                    self.last_checkpoint_time = time.time()

                # update rolling diagnostics occasionally
                if (self.iteration % self.nlive) == 0:
                    self.check_insertion_indices()

                auto_garbage_collect()

        except CheckPoint:
            self.checkpoint()
            sys.exit(130)

        # Refine evidence estimate
        self.logZ = self.finalise()
        self.info = self.integral_state.info
        # output the chain and evidence
        self.write_chain_to_file()
        self.write_evidence_to_file()
        if self.verbose:
            self.logger.critical('Final evidence: {0:0.2f}'.format(self.logZ))
            self.logger.critical('Information: {0:.2f}'.format(self.info))

        # Some diagnostics
        if self.verbose > 1 :
            self.integral_state.plot(os.path.join(self.output_folder,'logXlogL.png'))

        return 0

    def finalise(self):
        # final adjustments
        for i in range(self.nlive):
            self.integral_state.increment(self.logL[i], nlive=self.nlive-i, nreplace = 1)
            self.append_nested_sample(self.live_points[i], self.logP[i], self.logL[i])

        return self.integral_state.logZ
    
    def append_nested_sample(self, sample, logP, logL):
        x = np.zeros(1, dtype = {'names':self.output_names, 'formats':['f8' for _ in self.output_names]})
        for j, n in enumerate(self.model.names):
            x[n][0] = sample[j]
        x['logL'][0] = logL
        x['logPrior'][0] = logP
        self.nested_samples.append(x)
        
    def check_insertion_indices(self, rolling=True, filename=None):
        """
        Checking the distibution of the insertion indices either during
        the nested sampling run (rolling=True) or for the whole run
        (rolling=False).
        """
        if rolling:
            indices = np.array(self.insertion_indices)[-self.nlive:]
        else:
            indices = indices = np.array(self.insertion_indices)

        D, p = kstest(indices, 'uniform', args=(0, 1))
        if rolling:
            if self.verbose:
                self.logger.info('Rolling KS test: D={0:.3}, p-value={1:.3}'.format(D, p))
            self.rolling_p.append(p)
        else:
            if self.verbose:
                self.logger.info('Final KS test: D={0:.3}, p-value={1:.3}'.format(D, p))

        if filename is not None:
            np.savetxt(os.path.join(
                self.output_folder, filename),
                self.live_points.get_insertion_indices(),
                newline='\n',delimiter=' ')

    def get_output_folder(self):
        return self.output_folder

    def get_nested_samples(self):
        return self.nested_samples

    def get_prior_samples(self):
        return self.prior_samples

    def get_live_points(self):
        return self.live_points

    def get_logZ(self):
        return self.logZ

    def get_logLs(self):
        return self.logLs
    
    def get_logPs(self):
        return self.logPs
    
    def get_nlive(self):
        return self.nlive

    def get_information(self):
        return self.info

    def get_insertion_indeces(self):
        return self.insertion_indices
    
    def save(self):
        state = self.__dict__.copy()
        self.logger.critical('Saving Nested Sampling state in ' + str(self.resume_file))
        del state['model']
        del state['logger']

        with open(self.resume_file,'wb') as f:
            pickle.dump(state,f)

def get_worst(self, n):
    """
    selects the lowest likelihood N live points
    and updates the integral state
    """
    self.logLmin = self.logL[0]
    self.logLmax = self.logL[-1]
    self.worst = self._list[:n]
    self.nested_samples.extend(self.worst)
    self.state.increment(self.worst[n-1].logL, nreplace=self.n_replace)
    return self.worst

def replace_and_reinsert(live_points, logP, logL, idx, new_live_point, new_logP, new_logL):
    """
    arr  : (n, d) array
    keys : (n,) sorted in decreasing order
    idx  : index to replace
    """

    # Find new position
    insert_pos = np.searchsorted(logL, new_logL)

    # Account for removal of idx
    if insert_pos > idx:
        insert_pos -= 1

    # Shift block in-place
    if insert_pos < idx:
        live_points[insert_pos+1:idx+1] = live_points[insert_pos:idx]
        logP[insert_pos+1:idx+1] = logP[insert_pos:idx]
        logL[insert_pos+1:idx+1] = logL[insert_pos:idx]
    elif insert_pos > idx:
        live_points[idx:insert_pos] = live_points[idx+1:insert_pos+1]
        logP[idx:insert_pos] = logP[idx+1:insert_pos+1]
        logL[idx:insert_pos] = logL[idx+1:insert_pos+1]

    # Insert new data
    live_points[insert_pos] = new_live_point
    logL[insert_pos] = new_logL
    logP[insert_pos] = new_logP
    return (insert_pos - idx) / (len(live_points) - idx - 1)