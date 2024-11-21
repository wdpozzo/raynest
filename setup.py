import os
import platform
import numpy
from setuptools import setup, Extension, find_packages
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

setup(url="https://github.com/wdpozzo/raynest",  # Replace with your package URL
      packages=find_packages(),
      classifiers=[
                'Development Status :: 4 - Beta',
                'License :: OSI Approved :: MIT License',
                'Programming Language :: Python :: 3.9'
                'Programming Language :: Python :: 3.10'
                ],
      python_requires='>=3.9',
      ext_modules=ext_modules)

