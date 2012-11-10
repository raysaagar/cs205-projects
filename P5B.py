import numpy as np
import matplotlib.image as img
import time

# Import the PyCUDA modules
import pycuda.compiler as nvcc
import pycuda.gpuarray as gpu
import pycuda.driver as cu
from pycuda.reduction import ReductionKernel
from pycuda.elementwise import ElementwiseKernel

# Initialize the CUDA device
import pycuda.autoinit

# Define CUDA kernel as a string

front_kernel_source = \
"""
// initial front generator

__global__ void front_kernel(float* im, char* seeds, int* old_im, int height, int width, float start, float end)
{
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  // x is j, y is i REMOVE

  int tid = y*width + x;

  if (x < width && y < height){
    if(im[tid] >= start && im[tid] <= end){
      seeds[tid] = 1;
      if(y > 0)
        old_im[(y-1)*width + x] = (y-1)*width + x;
      if(y < (height - 1))
        old_im[(y+1)*width + x)] = (y+1)*width + x;
      if(x > 0)
        old_im[tid-1] = tid-1;
      if(x < (width - 1))
        old_im[tid+1] = tid+1;
    }
  }
}
"""
new_front_kernel_source = \
"""
__global__ void new_front_kernel(float* im, char* seeds, int* old_im, int* new_im, int height, int width, int sz, float start, float end)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;

  if (x < (height*sz) && tid < sz){
    int index = old_im[tid];

    if(index != -1 && seeds[index] == 0 && im[index] >= start && im[index] <= end){
      seeds[index] = 1;

      int a = index/width;
      int b = index % width;
      //a is i, b is j REMOVE

      if (a < height && b < width){
        if(a > 0)
          new_im[index-width] = index-width
        if(a < height - 1)
          new_im[index+width] = index+width;
        if(b > 0)
          new_im[index-1] = index-1;
        if(b < width - 1)
          new_im[index+1] = index+1;
      }
    }
  }
}
"""

# Image files
in_file_name  = "Harvard_Medium.png"
out_file_name = "Harvard RegionGrow GPU B.png"

# Region growing constants [min, max]
seed_threshold = [0, 0.08];
threshold      = [0, 0.27];

def cuda_compile(source_string, function_name):
  # Compile the CUDA Kernel at runtime
  source_module = nvcc.SourceModule(source_string)
  # Return a handle to the compiled CUDA kernel
  return source_module.get_function(function_name)

if __name__ == '__main__':
  # compile kernels
  front_kernel = cuda_compile(front_kernel_source,"front_kernel")
  new_front_kernel = cuda_compile(new_front_kernel_source,"new_front_kernel")

  # Read image. BW images have R=G=B so extract the R-value
  image = img.imread(in_file_name)[:,:,0]
  height, width = np.int32(image.shape)
  print "Processing %d x %d image" % (width, height)

  tpbx = 32
  tpby = 8
  nBx = int(width/tpbx)
  nBy = int(height/tpby)
  blocksize = (tpbx,tpby,1)
  gridsize = (nBx, nBy)

  threads_q = tbpx*tbpy
  bxq = int(np.ceil(1.0*height*width/threads_q))
  blocksize_q = (threads_q, 1,1)
  gridsize_q = (bxq, 1)


  im = np.array(image, dtype=np.float32)
  queue = np.int32(np.zeros(height*width)-1)
  im_reg = np.int8(np.zeros([height,width]))
  queuelen = 0

  #gpu timers
  startT = cu.Event()
  endT = cu.Event()

  transfertime = time.time()
  im_d = gpu.to_gpu(im)
  seeds_d = gpu.to_gpu(im_reg)
  old_q_d = gpu.to_gpu(queue)
  new_q_d = gpu.to_gpu(queue)

  total_time = time.time() - transfertime

  # run the kernel
  startT.record()
  front_kernel(im_d, seeds_d, old_q_e, height, width, np.float32(seed_threshold[0]), np.float32(seed_threshold[1]), block=blocksize, grid=gridsize)
  endT.record()
  endT.synchronize()
  gpuT = startT.time_till(endT) * 1e-3

  transfertime = time.time()
  old_q = old_q_d.get()
  unique_q = np.unique(old_q)
  unique_q = np.delete(unique_q,0)
  old_q_d = gpu.to_gpu(unique_q)
  queuelen = len(unique_q)
  gridsize_q = (int(np.ceil(1.0*queuelen/threads_q)),1)

  total_time += time.time() - transfertime

  iter = 0

  while queuelen != 0:
    startT.record()
    new_front_kernel(im_d, seeds_d, old_q_d, new_q_d, height, width, np.int32(queuelen), np.float32(threshold[0]), np.float32(threshold[1]), block=blocksize_q, grid=gridsize_q)
    endT.record()
    endT.synchronize()
    gpuT += startT.time_till(endT) * 1e-3
    transfertime = time.time()
    new_q = new_q_d.get()
    unique_q = np.unique(old_q)
    unique_q = np.delete(unique_q,0)
    old_q_d = gpu.to_gpu(unique_q)
    new_q_d = gpu.to_gpu(queue)
    queuelen = len(unique_q)
    gridsize_q = (int(np.ceil(1.0*queuelen/threads_q)),1)
    total_time += time.time() - transfertime
    iter = iter+1

  # save image, clamp values
  transfertime = time.time()
  finalim = seeds_d.get()
  total_time += time.time() - transfertime
  img.imsave(out_file_name, finalim, cmap='gray', vmin=0, vmax=1)

  print "Iterations: %d" % iter
  print "GPU Time: %f" % gpuT
  print "Total Time: %f" % (total_time + gpuT)


