import warnings
import numpy as np
import raynest.model
from scipy.stats import norm
from scipy.special import logsumexp

def hypertriangulate(x, bounds=(0, 1)):
    """
    Transform a vector of numbers from the hypercube to the hypertriangle.

    The hypercube is the space the samplers usually work in; the 
    components of x are in no particular order.
    
    The hypertriangle is the space where the components are sorted into
    ascenting order, \ ( y0 < y1 < ... < yn \ ). 
    
    The (unit) transformation is defined by:

    .. math::
        y_j = 1 - \\prod_{i=0}^{j} (1 - x_i)^{1/(n-i)}

    Example application. If we are analysing a number num_dim of DWD 
    sources, all with identical priors. Then this function would be
    called on the array `np.array([f_1, f_2, ..., f_num_sources])` with
    `bounds=(f_min, f_max)`.

    Parameters
    ----------
    x: array
        The hypercube parameter values. 
        The components of x are in no particular order.
        Input shape = (num_dim,) or (num_points, num_dim).
        If input array is multi-dimensional, the function is vectorised 
        along all except the last axis.

    bounds: tuple
        Lower and upper bounds of parameter space. Default is to transform
        between the unit hypercube and unit hypertriangle with (0, 1).

    Returns
    -------
    y: array, shaped like x
        The hypertriangle parameter values
        The components of y are sorted in ascending order.
    """
    x = np.array(x)
    
    # transform to the unit hypercube
    unit_x = (x - bounds[0]) / (bounds[1] - bounds[0])

    # hypertriangle transformation
    with warnings.catch_warnings():
        # this specific warning is raised when unit_x goes outside [0, 1]
        warnings.filterwarnings('error', 'invalid value encountered in power')
        try:
            n = np.size(unit_x, axis=-1)
            index = np.arange(n)
            inner_term = np.power(1 - unit_x, 1/(n - index))
            unit_y = 1 - np.cumprod(inner_term, axis=-1)
        except RuntimeWarning:
            raise ValueError('Values outside bounds passed to hypertriangulate')

    # re-apply orginal scaling, offset
    y = bounds[0] + unit_y * (bounds[1] - bounds[0])

    return y

def gmm(x, weights, means, stds):
    """
    Vectorized 1D Gaussian Mixture Model.
    
    Args:
        x : array of shape (N,) - data points
        weights : array of shape (K,) - mixture weights (must sum to 1)
        means : array of shape (K,) - means of Gaussians
        stds : array of shape (K,) - std deviations of Gaussians (positive)
    
    Returns:
        float : log-likelihood of the dataset
    """
    x = np.atleast_1d(x)
    weights = np.asarray(weights)
    means = np.asarray(means)
    stds = np.asarray(stds)

    if not np.isclose(np.sum(weights), 1.0):
        raise ValueError("Mixture weights must sum to 1.")
    if np.any(stds <= 0):
        raise ValueError("Standard deviations must be positive.")

    # Shape (N, K): log pdf of each Gaussian at each x
    log_probs = norm.logpdf(x[:, None], loc=means[None, :], scale=stds[None, :])

    # Add log weights (broadcasted)
    log_weighted = log_probs + np.log(weights[None, :])

    # Log-sum-exp across mixture components (axis=1 = sum over K)
    log_mixture = logsumexp(log_weighted, axis=1)
    
    return log_mixture

def gmm_log_likelihood(x, weights, means, stds):
    """
    Vectorized log-likelihood for a 1D Gaussian Mixture Model.
    
    Args:
        x : array of shape (N,) - data points
        weights : array of shape (K,) - mixture weights (must sum to 1)
        means : array of shape (K,) - means of Gaussians
        stds : array of shape (K,) - std deviations of Gaussians (positive)
    
    Returns:
        float : log-likelihood of the dataset
    """
#    x = np.atleast_1d(x)
#    weights = np.asarray(weights)
#    means = np.asarray(means)
#    stds = np.asarray(stds)
#
#    if not np.isclose(np.sum(weights), 1.0):
#        raise ValueError("Mixture weights must sum to 1.")
#    if np.any(stds <= 0):
#        raise ValueError("Standard deviations must be positive.")
#
#    # Shape (N, K): log pdf of each Gaussian at each x
#    log_probs = norm.logpdf(x[:, None], loc=means[None, :], scale=stds[None, :])
#
#    # Add log weights (broadcasted)
#    log_weighted = log_probs + np.log(weights[None, :])

    # Log-sum-exp across mixture components (axis=1 = sum over K)
    log_mixture = gmm(x, weights, means, stds)

    # Total log-likelihood
    return np.sum(log_mixture)

class GaussianMixtureModel(raynest.model.Model):
    """
    A simple gaussian model with parameters mean and sigma
    Shows example of using your own data
    """
    def __init__(self, n, data):
        self.n = n
        self.data = data
        # append all parameters
        self.names=[]
        self.bounds=[]
        
        for i in range(self.n):
            self.names.append('mean_{}'.format(i))
            self.bounds.append([-5,5])
            self.names.append('sigma_{}'.format(i))
            self.bounds.append([0.0,5.0])
        if self.n > 1:
            for i in range(self.n):
                self.names.append('w_{}'.format(i))
                self.bounds.append([0.0,1.0])

    def log_likelihood(self, x):
        ## apply hypertriangulation by ordering in the mean

        # extract, for all the components, all the means (the copy should prevent the memory address
        # of the samples sampled from the hypercube to be changed, so that the sampler thinks to sample from it)
        _x = x.copy()
        mean_names = ['mean_{}'.format(j) for j in range(self.n)]
        means_vector = np.array([_x[name] for name in mean_names])

        # hypertriangulate
        means_vector = hypertriangulate(means_vector,
                                        bounds=(self.bounds[0][0], self.bounds[0][1])) # should be the bounds of the mean

        # order the means in all the components
        for i in range(len(mean_names)):
            _x[mean_names[i]] = means_vector[i]

        # call gmm log likelihood
        if self.n > 1:
            w = [_x['w_{}'.format(j)] for j in range(self.n)]
        else:
            w = [1.0]
        #m = [x['mean_{}'.format(j)] for j in range(self.n)]
        s = [_x['sigma_{}'.format(j)] for j in range(self.n)]

        # means_vector, w, s should be already correctly ordered
        return gmm_log_likelihood(self.data, w, means_vector, s)

    def log_prior(self,p):

        if self.n > 1:
            #set the last weight so that we live on the simplex
            p['w_{}'.format(self.n-1)] = 1.0 - np.sum([p['w_{}'.format(j)] for j in range(self.n-1)])

        return super().log_prior(p)

if __name__=='__main__':

    import h5py
    import corner
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    
    postprocess = 1
    product_space = range(1,10)
    rng = np.random.default_rng(42)
    data = np.concatenate([rng.normal(2, 0.05, 100), rng.normal(3, 1.0, 600), rng.normal(0, 0.4, 300)])
    
    xaxis          = np.linspace(-10,10,1001)
    mixture_models = np.zeros((xaxis.shape[0],len(product_space),3))
    logZ           = np.zeros(len(product_space))
    dlogZ          = np.zeros(len(product_space))
    
    true_mixture   = gmm(xaxis,[1./10.,6./10.,3./10.],[2.0,3.0,0.0],[0.05,1.0,0.4])
    samples        = []
    
    for it,working_hypothesis in enumerate(product_space):
        
        model=GaussianMixtureModel(working_hypothesis, data)
        
        if not(postprocess):
        
            work=raynest.raynest(model, verbose=2,
                               nnest=3, nensemble=3, nlive=500, maxmcmc=5000, nslice=0, nhamiltonian=0, seed = 1,
                               resume=0, periodic_checkpoint_interval=600, output='./mixture_model_{}'.format(working_hypothesis))
            work.run()
    
#    parent_dir = "/mnt/c/Users/Gabriele/Documenti/PhD/CTMCMC/"
 
   
        base_name = ['mean', 'sigma', 'w']
        n_samples = 0
     
        with h5py.File("mixture_model_{}/raynest.h5".format(working_hypothesis), 'r') as fo:
        
            post_samples = fo['combined'].get('posterior_samples')[()]
            logZ[it] = fo['combined'].get('logZ')[()]
            dlogZ[it] = fo['combined'].get('logZ_error')[()]
            n_samples = len(post_samples)
     
#            samples = np.zeros((n_samples, working_hypothesis, len(base_name)))
            
#            for i in range(n_samples):

#            for k,name in enumerate(base_name):
#                for j in range(working_hypothesis):
#                    print(it,j,k,post_samples[name+f"_{k}".format(k)],name+f"_{k}".format(k))
#                    samples[:,j,k] = post_samples[name+f"_{k}".format(k)]
     
#        samples = np.array(samples).reshape((n_samples, working_hypothesis, len(base_name)))
        
        means = np.array([post_samples[f"mean_{k}"] for k in range(working_hypothesis)]).T
        sigmas = np.array([post_samples[f"sigma_{k}"] for k in range(working_hypothesis)]).T
        weights = np.array([post_samples[f"w_{k}"] for k in range(working_hypothesis)]).T if working_hypothesis > 1 else np.ones((n_samples,1))
        
        samples = np.row_stack((means,sigmas,weights)).T
        mixture_model = np.zeros((xaxis.shape[0],n_samples))
    
        for i in tqdm(range(n_samples),desc=f'components = {working_hypothesis} logZ = {logZ[it]}'):
            
            mean_vector = hypertriangulate([m for m in means[i]], bounds=(model.bounds[0][0], model.bounds[0][1]))
            mixture_model[:,i] = np.exp(gmm(xaxis,[w for w in weights[i]],mean_vector,[s for s in sigmas[i]]))
        
        mixture_models[:,it,:] = np.percentile(mixture_model,[5,50,95],axis=1).T

    print(dlogZ)
    cmap = plt.cm.tab10
    colors = cmap(np.linspace(0, 1, len(product_space)))
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xaxis,np.exp(true_mixture),color='k',lw=0.75,linestyle='dashed')
    for it,working_hypothesis in enumerate(product_space):
        ax.plot(xaxis,mixture_models[:,it,1],color=colors[it],lw=0.75)
        ax.fill_between(xaxis,mixture_models[:,it,0],mixture_models[:,it,2],facecolor=colors[it],alpha=0.5)
    
    ax.set_xlabel('x')
    ax.set_ylabel('p(x)')
    plt.savefig("reconstructed_gmm.pdf",bbox_inches='tight')
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i,c in enumerate(colors):
        ax.errorbar(product_space[i],logZ[i]-np.max(logZ),yerr=dlogZ[it],color=c,lw=0.75,ls=None,marker='o',ms=6)
    ax.axvline(3,linestyle='dashed',color='r')
    ax.grid(alpha=0.5,linestyle='dotted')
    ax.set_xlabel('# components')
    ax.set_ylabel('logB relative to the maximum')
    plt.savefig("logB_gmm.pdf",bbox_inches='tight')
    plt.show()
    
#        # correct here
#        for i in range(working_hypothesis):
#            corner.corner(samples[:,i,:])
#        plt.show()
