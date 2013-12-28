import numpy as np
import time

# Import the PyCUDA modules
import pycuda.compiler as nvcc
import pycuda.gpuarray as gpu
import pycuda.driver as cu
# Initialize the CUDA device
import pycuda.autoinit

if __name__ == '__main__':

  # On the host, define the vectors, be careful about the types
  N      = np.int32(2**18)
  m = 1000
  z      = np.float32(np.zeros(N))
  alpha  = np.float32(100.0)
  x      = np.float32(np.random.random(N))
  y      = np.float32(np.random.random(N))

  starttime = time.time()
  for i in xrange(N):
    z[i] = alpha*x[i] + y[i]
  endtime = time.time()

  bytes = N*4
  totaltime = endtime-starttime

  print "bytes: %d" % bytes
  print "total time: %f s" % totaltime
  print "time per byte (usec): %f" % (totaltime * 1e6 / bytes)
