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

# Define the CUDA saxpy kernel as a string.
min_kernel_source = \
'''
__global__ void min_kernel(float* d, float* d_min)
{
  // want to use shared memory
  extern __shared__ float s[];

  int x = threadIdx.x;
  // copy into shared memory
  s[x] = d[x];

  //BUG: We want to sync threads here to make sure
  //       that all of the s data is copied first
  __syncthreads();

  int half = blockDim.x/2;
  while(half > 0){
    
    //BUG: we want to make sure all threads will sync the same number of times
    //     we do not want to have zombie threads
    if (x < half && s[x] > s[x+half])
      s[x] = s[x + half];
    
    half = half/2;
    
    //BUG: we want to sync here so all reductions happen after all threads finish
    __syncthreads();

  }

  // Only the first thread should return, avoids communication overhead
  if(x == 0)
    d_min[0] = s[0];

}
'''

def cuda_compile(source_string, function_name):
  source_module = nvcc.SourceModule(source_string)
  return source_module.get_function(function_name)


if __name__ == '__main__':
  min_kernel = cuda_compile(min_kernel_source, "min_kernel")
  i = 0
  blocksize = (2**10, 1, 1)
  gridsize = (1, 1)
  while True:
    N = np.int32(2**10)
    d_data = np.float32(np.random.random(N))
    d_min  = np.float32(np.zeros(1))

    d_data_d = gpu.to_gpu(d_data)
    d_min_d = gpu.to_gpu(d_min)

    min_kernel(d_data_d, d_min_d, block = blocksize, grid = gridsize, shared = d_data_d.nbytes)

    d_min_gpu = d_min_d.get()

    if(d_min_gpu[0] == np.min(d_data)):
      print "Correct %d" % i
      i = i + 1
    else:
      print "Failed! (%d)" % i
      break






