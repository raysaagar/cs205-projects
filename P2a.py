import numpy as np
import time

# Import the PyCUDA modules
import pycuda.compiler as nvcc
import pycuda.gpuarray as gpu
import pycuda.driver as cu
# Initialize the CUDA device
import pycuda.autoinit

# Define the CUDA saxpy kernel as a string.
saxpy_kernel_source = \
"""
__global__ void saxpy_kernel(float* z, float alpha, float* x, float* y, int N)
{
    // HW3 P2: WRITE ME
    // get TID
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if (tid < N) z[tid] = alpha*x[tid] + y[tid];

}
"""

def cuda_compile(source_string, function_name):
  # Compile the CUDA Kernel at runtime
  source_module = nvcc.SourceModule(source_string)
  # Return a handle to the compiled CUDA kernel
  return source_module.get_function(function_name)


def bandwidth():
#  n = 11
  n = 1
  nbytes = np.zeros(n, dtype = np.float64)
  time = np.zeros(n, dtype = np.float64)
  Time = np.zeros(n, dtype = np.float64)
  
  timecputogpu = []
  timegputocpu = []

  for b in range(n):
    #nbytes[b] = int(8**b)
    nbytes[b] = int(2**18)
    data = np.random.random(nbytes[b]/8)
    start_time = cu.Event()
    end_time = cu.Event()

    start_time.record()
    data_d = gpu.to_gpu(data)
    end_time.record()
    end_time.synchronize()

    timecputogpu.append(start_time.time_till(end_time) * 1e-3)
    
    start_time.record()
    data_gpu = data_d.get()
    end_time.record()
    end_time.synchronize()

    timegputocpu.append(start_time.time_till(end_time) * 1e-3)

    print "size = %10.d bytes, transfer time = %f s, %f s" % (nbytes[b], timecputogpu[b],timegputocpu[b])

  Latc2g = (sum(timecputogpu[:4]) / 4.0) * 1e6
  Latg2c = (sum(timegputocpu[:4]) / 4.0) * 1e6

  Bandc2g = (((nbytes[n-1] - nbytes[n-2]) / (timecputogpu[n-1] - timecputogpu[n-2])) / 1e6)
  Bandg2c = (((nbytes[n-1] - nbytes[n-2]) / (timegputocpu[n-1] - timegputocpu[n-2])) / 1e6)

  print "CPU TO GPU"
  print "\n Latency   = %f usec" % Latc2g
  print " Bandwidth = %f MBytes/sec" % Bandc2g

  print "CPU TO GPU"
  print "\n Latency   = %f usec" % Latg2c
  print " Bandwidth = %f MBytes/sec" % Bandg2c



if __name__ == '__main__':
  
  #calculate bandwidth and latency
  bandwidth()

  # Compile the CUDA kernel
  saxpy_kernel = cuda_compile(saxpy_kernel_source,"saxpy_kernel")

  # On the host, define the vectors, be careful about the types
  N      = np.int32(2**18)
  m = 1000
  z      = np.float32(np.zeros(N))
  alpha  = np.float32(100.0)
  x      = np.float32(np.random.random(N))
  y      = np.float32(np.random.random(N))

  # On the host, define the kernel parameters - for PART 1, modify and time these!
  blocksize = (8,1,1)     # The number of threads per block (x,y,z)
  gridsize  = (32768,1)   # The number of thread blocks     (x,y)

  # Allocate device memory and copy host to device
  x_d = gpu.to_gpu(x)
  y_d = gpu.to_gpu(y)
  z_d = gpu.to_gpu(z)

  # Initialize the GPU event trackers for timing
  start_gpu_time = cu.Event()
  end_gpu_time = cu.Event()

  # Run the CUDA kernel with the appropriate inputs
  start_gpu_time.record()
  for i in xrange(1000):
    saxpy_kernel(z_d, alpha, x_d, y_d, N, block=blocksize, grid=gridsize)
  end_gpu_time.record()
  end_gpu_time.synchronize()
  gpu_time = start_gpu_time.time_till(end_gpu_time) * 1e-3

  print "GPU Time: %f" % (gpu_time/1000)
  print "Blocksize: %f" % blocksize[0]
  # Copy from device to host
  z_gpu = z_d.get()

  # Compute the result in serial on the host
  start_serial_time = time.time()
  z_serial = alpha * x + y
  end_serial_time = time.time()
  print "Serial Time: %f" % (end_serial_time - start_serial_time)

  # Compute the error between the two
  rel_error = np.linalg.norm(z_gpu - z_serial) / np.linalg.norm(z_serial)

  # Print error message
  if rel_error < 1.0e-5:
    print "Hello CUDA test passed with error %f" % rel_error
  else:
    print "Hello CUDA test failed with error %f" % rel_error
