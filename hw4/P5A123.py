import numpy as np
import matplotlib.image as img
import time

# Import PyCUDA modules
import pycuda.compiler as nvcc
import pycuda.gpuarray as gpu
import pycuda.driver as cu

# Initialize the CUDA device
import pycuda.autoinit

in_file_name = "Harvard_Small.png"
out_file_name = "Harvard_GrowRegion_CPU_CUDA_Small.png"

# Region growing constants [min, max]
#seed_threshold = [0, 0.08];
#threshold      = [0, 0.27];
seed_threshold = np.float32(0.08)
threshold = np.float32(0.27)

filter_source = \
"""

#define getflag(x,y)        d_data[width * (y) + (x)]

__global__ void filter_kernel(int* d_data, float* p_data, int* z, int width, int height, float threshold) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
	int tid = width*(y) + x;

  if (d_data[tid])
	z[tid] = 1;
  else if (p_data[tid] > threshold)
	z[tid] = 0;
  else if (x > 0 && getflag(x-1, y)) 
	z[tid] = 1;
  else if (y > 0 && getflag(x, y-1)) 
	z[tid] = 1;
  else if (x < width - 1 && getflag(x+1, y)) 
	z[tid] = 1;
  else if (y < height- 1 && getflag(x, y+1)) 
	z[tid] = 1;
  else
	z[tid] = 0;

}
"""

def cuda_compile(source_string, function_name):
  # Compile the CUDA kernel at runtime
  source_module = nvcc.SourceModule(source_string)
  return source_module.get_function(function_name)

filter_kernel = cuda_compile(filter_source, "filter_kernel")

if __name__ == '__main__':
  # Read image. BW images have R=G=B so extract the R-value
  image = img.imread(in_file_name)[:,:,0]
  height, width = np.int32(image.shape)
  area = height*width
  print "Processing %d x %d image" % (width, height)

  blocksize = (32,32,1)
  gridsize = (int(width/32),int(height/32))

  ones = np.empty([height,width])
  ones[:,:] = 1

  im_d = gpu.to_gpu(np.float32(np.array(image)))
  ones_d = gpu.to_gpu(np.int32(np.array(ones)))
  zero_d = gpu.to_gpu(np.int32(np.zeros([height,width])))
  threshold_d = gpu.if_positive(im_d - seed_threshold, zero_d, ones_d)
  new_d = gpu.to_gpu(np.int32(np.zeros([height,width])))

  old_flags = 0
  new_flags = gpu.sum(threshold_d).get()
  
  while (new_flags - old_flags) != 0:
    old_flags = new_flags

    # run filter kernel
    filter_kernel(threshold_d, im_d, new_d, width, height, threshold, block=blocksize, grid=gridsize)

    new_flags = gpu.sum(new_d).get()

    # transfer output to input
    threshold_d = new_d

  img.imsave(out_file_name, threshold_d.get(), cmap='gray',vmin=0,vmax=1)

