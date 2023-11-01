from setuptools import setup, find_packages, Extension
from distutils.core import setup
from Cython.Build import build_ext, cythonize

import numpy as np

# To build the module call:
#  python setup.py build_ext --inplace 

ext_modules = [
        Extension("orthpoly",
                  sources=["orthpoly.pyx",
                           "OrthogonalPolynomialSlidingWindow.cpp",
                           ],
                  language="c++",
                  include_dirs=[np.get_include()]
                  )
        ]


setup(name='orthpoly',
      packages=find_packages(),
      cmdclass={'build_ext': build_ext},
      ext_modules=cythonize(ext_modules),
     )
