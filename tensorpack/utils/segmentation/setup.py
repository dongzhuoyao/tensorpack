# Author: Tao Hu <taohu620@gmail.com>

# setup.py
from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy
setup(ext_modules = cythonize(Extension(
    'fastUpdateConfusionMatrix',
    sources=['fastUpdateConfusionMatrix.pyx'],
    language='c',
    include_dirs=[numpy.get_include()],
    library_dirs=[],
    libraries=[],
    extra_compile_args=[],
    extra_link_args=[]
)))

#python setup.py build_ext --inplace