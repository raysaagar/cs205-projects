import numpy as np
import time
import math
import matplotlib.pyplot as plt

# Import the PyCUDA modules
import pycuda.compiler as nvcc
import pycuda.gpuarray as gpu
import pycuda.driver as cu

# Initialize the CUDA device
import pycuda.autoinit

# Define the CUDA kernel
D2x_kernel_source = \
"""
__global__ void D2x_kernel(double* b, double* a, int N, double dx)
{
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid > 0 && tid < N-1)
    a[tid] = (a[tid-1] - 2*a[tid] + a[tid+1]) / (dx*dx);
}
"""
def cuda_compile(source_string, function_name):
  source_module = nvcc.SourceModule(source_string)
  return source_module.get_function(function_name)

if __name__ == '__main__':
  # Compile the CUDA kernel
  D2x_kernel = cuda_compile(D2x_kernel_source, "D2x_kernel")

  N      = np.int32(2**20)
  x      = np.float64(np.linspace(0, 1, 2**20))
  a      = np.float64(np.array([math.sin(i) for i in x]))
  b      = np.float64(np.empty(2**20))
  dx     = np.float64(1./(N-1))

  a_d = gpu.to_gpu(a)
  b_d = gpu.to_gpu(b)

  blocksize = (512,1,1)
  gridsize = (2048,1)

  D2x_kernel(b_d,a_d,N,dx,block=blocksize,grid=gridsize)

  # copy data back
  b_gpu = a_d.get()

  # print out to verify
  # print b_gpu

  # generate graphs - thanks to Will Chen, Sebastian Chiu for graphing help
  plt.figure()
  plt.title('Sine Curve w/ 2nd Derivative')
  plt.xlabel('x')
  plt.ylabel('sin(x) approximation')
  plt.legend(('2nd Deriv. Approx.','sin(x) curve'), loc=2)
  plt.plot(x[2:-1],b_gpu[2:-1])
  plt.plot(x,a)
  plt.show()

