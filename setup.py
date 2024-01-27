import os
import platform
import numpy
from setuptools import setup, Extension
from Cython.Build import cythonize

WINDOWS = platform.system().lower() == "windows"

# set extension
libraries = [] if WINDOWS else ["m"]
ext_modules = [Extension("raynest.parameter",
                         sources=[os.path.join("raynest", "parameter.pyx")],
                         include_dirs=['raynest', numpy.get_include()],
                         libraries=libraries,
                         extra_compile_args=["-O3","-ffast-math","-mavx2","-ftree-vectorize"])]
ext_modules = cythonize(ext_modules)
setup(ext_modules=ext_modules)

