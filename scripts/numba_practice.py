"""Sending functions to a GPU.

Created on Tue Dec  7 14:39:40 2021

@author: travis
"""
import numba as nb
import numpy as np
import math

from human_byte_size import hbsize
from numba import cuda, types
from numba.typed import Dict


# Basic Function
def fun(a, b):
    return math.sin(a) * math.cos(b)


# Numpy's vecotrized function
npfun = np.vectorize(fun)


# Simple, does not compile until it receives its first inputs because it
# doesn't know the appropriate input and output datatypes. This makes the first
# function call somewhat slower than subsequent ones.
@nb.vectorize()
def nbfun1(a, b):
    return math.sin(a) * math.cos(b)


# Avoids using python runtime.
@nb.vectorize(nopython=True)
def nbfun2(a, b):
    return math.sin(a) * math.cos(b)


# This compiles right when it is defined (eage compilation) given a list of
# options for input and output datatypes. The first run of a function will be
# quicker since it is precompiled. Format: output_type(input_type, input_type).
@nb.vectorize(["f8(i8,i8)", "f4(f4,f4)", "f8(i8,i8)"], nopython=True)
def nbfun3(a, b):
    return math.sin(a) * math.cos(b)


# If you send this function to your nvidia gpu, it appears to automatically
# turn the python runtime off. There is a deprecation warning about eager
# comilation, but the signature which triggers it is still apparently
# required.
@nb.vectorize(["f8(i8,i8)", "f4(f4,f4)", "f8(i8,i8)"], target="cuda")
def nbfun5(a, b):
    return math.sin(a) * math.cos(b)


# Same deal as above i think
@nb.vectorize(["f8(i8,i8)", "f4(f4,f4)", "f8(i8,i8)"], target="parallel")
def nbfun4(a, b):
    return math.sin(a) * math.cos(b)


# Is it possible to map a dictionary like object to an array?
def map_test():
    """Map dictionary values to an array of keys."""
    # int64 = nb.int64[:]
# "void(int64[:],int64[:])"
    @cuda.jit
    def mapit(dct, key):
        """Map key value pairs in d to a."""
        key = dct[key]

    # initialize a typed numba dictionary
    dct = Dict.empty(
        key_type=types.int64,
        value_type=types.int64
    )

    # Say you have a dictionary
    for i in range(1_000):
        value = np.random.randint(1_001, 2_000, 1)[0]
        dct[i] = value

    # Create a 2D array out of the dictionary
    dct = np.array([list(dct.keys()), list(dct.values())])

    # Create an array of keys
    keys = np.random.randint(0, 1_000, 10_000).reshape(100, 100)
    mapit(dct, [999, 998])

    


if __name__ == "__main__":
    a = np.random.randint(0, 100, 500_000_000)
    print(hbsize(a))
    b = 5
    # %timeit -n 1 -r 1 x = nbfun4(a, b)
    # %timeit -n 1 -r 1 x = nbfun5(a, b)
    