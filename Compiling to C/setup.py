from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name='CythonDemo',
    ext_modules=cythonize(["cython1.pyx", "cython2.pyx", "cython3.pyx", "cython4.pyx"]),
    include_dirs=[np.get_include()]
)
