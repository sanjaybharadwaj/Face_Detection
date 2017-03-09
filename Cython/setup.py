from distutils.core import setup
from Cython.Build import cythonize

setup(
	ext_modules = cythonize("Haar_cascade_detection.pyx",
                                language="c++",
))
