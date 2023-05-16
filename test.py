from numba import cuda
import numpy as np

@cuda.jit
def print_size(arr,result):
    result[0]=arr.shape[0]

arr = np.array([1, 2, 3, 4, 5])
result = np.array([0])
print_size[1, 1](arr,result)
print(result)