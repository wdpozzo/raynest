import logging
from .utils import StreamHandler, LEVELS
# Configure base logger for the raynest package - inherited by all loggers with
# names prefixed by 'raynest'
logger = logging.getLogger('raynest')
logger.setLevel(LEVELS[-1])  # maximum verbosity recorded to base logger
logger.addHandler(logging.NullHandler())

console_handler = StreamHandler(verbose=0)  # default console verbosity is 0
logger.addHandler(console_handler)
# To change the console handler verbosity:
#   from raynest import console_handler
#   console_handler.set_verbosity(2)
__version__="1.0.2"

from .raynest import raynest

# Get the version number from git tag
from pkg_resources import get_distribution, DistributionNotFound
try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    __version__ = "dev"

__all__ = ['model',
           'NestedSampling',
           'parameter',
           'sampler',
           'raynest',
           'nest2pos',
           'proposal',
           'plot',
           'logger']
