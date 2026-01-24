from setuptools import setup, Extension
import pybind11

module = Extension(
    'incomplete_gamma',  # This will be the Python module name
    sources=['incomplete_gamma.cpp'],  # C++ file name
    include_dirs=[
        pybind11.get_include(),
        '/opt/homebrew/opt/gmp/include',  # GMP include path
        '/opt/homebrew/opt/mpfr/include',  # MPFR include path
    ],  
    libraries=['mpfr', 'gmp'],  # Libraries to link against
    library_dirs=[
        '/opt/homebrew/opt/gmp/lib',  # GMP lib path
        '/opt/homebrew/opt/mpfr/lib',  # MPFR lib path
    ],  
)

setup(
    name='incomplete_gamma',
    ext_modules=[module],
)
