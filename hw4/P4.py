import numpy as np
import matplotlib.image as img
import time
# Import the PyCUDA modules
import pycuda.compiler as nvcc
import pycuda.gpuarray as gpu
import pycuda.driver as cu

# Initialize the CUDA device
import pycuda.autoinit

# Define the CUDA kernel as a string.
sharpen_kernel_source = \
"""
__global__ void sharpen_kernel(float* curr_im, float* next_im, int height, int width, float epsilon)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int gid = width*i + j;

  if (i > 0 && j > 0 && i < height-1 && j < width-1){
    next_im[gid] = curr_im[gid] + epsilon * (
      -1*curr_im[gid-width-1] + -2*curr_im[gid-1] + -1*curr_im[gid+width-1]
      + -2*curr_im[gid-width] + 12*curr_im[gid] + -2*curr_im[gid+width]
      + -1*curr_im[gid-width+1] + -2*curr_im[gid+1] + -1*curr_im[gid+width+1]);
  }
}
"""

# Image files
in_file_name  = "Harvard_Small.png"
out_file_name = "Harvard_Sharpened_CUDA_CPU_Small.png"
# Sharpening constant
EPSILON    = np.float32(.005)

def cuda_compile(source_string, function_name):
  # Compile the CUDA Kernel at runtime
  source_module = nvcc.SourceModule(source_string)
  # Return a handle to the compiled CUDA kernel
  return source_module.get_function(function_name)

def mean_variance(data):
  '''Return the mean and variance of a 2D array'''
  mean = np.sum(data) / data.size
  variance = np.sum(np.square(data - mean)) / data.size
  print "Mean = %f,  Variance = %f" % (mean, variance)
  return mean, variance

if __name__ == '__main__':
  
  # Compile the CUDA kernel
  sharpen_kernel = cuda_compile(sharpen_kernel_source,"sharpen_kernel")

  # Read image. BW images have R=G=B so extract the R-value
  original_image = img.imread(in_file_name)[:,:,0]

  # Get image data
  height, width = np.int32(original_image.shape)
  print "Processing %d x %d image" % (width, height)

  # Allocate memory
  curr_im, next_im = np.array(original_image), np.array(original_image)
  # Compute the image's initial mean and variance
  init_mean, init_variance = mean_variance(curr_im)
  variance = init_variance

  # define the kernel parameters
  TPBx = 32
  TPBy = 32
  nBx = int(height/TPBx)
  nBy = int(width/TPBy)
  blocksize = (nBx,nBy,1)     # The number of threads per block (x,y,z)
  gridsize = (TPBx,TPBx)        # The number of thread block (x,y)
  #blocksize = (256,256,1)
  #gridsize = (256,1)

  while variance < 1.1 * init_variance:

    # Allocate device memory and copy host to device
#    height_d = gpu.to_gpu(height)
#    width_d = gpu.to_gpu(width)
    curr_im_d = gpu.to_gpu(curr_im)
    next_im_d = gpu.to_gpu(next_im)
#    epsilon_d = gpu.to_gpu(EPSILON)
    sharpen_kernel(curr_im_d, next_im_d, height,width,EPSILON, 
      block = (TPBx,TPBy,1), grid=(nBx,nBy))

#  while variance < 1.1 * init_variance:
    # Compute Sharpening
#    for i in range(1, height-1):
#      for j in range(1, width-1):
#        next_im[i,j] = curr_im[i,j] + EPSILON * (
#             -1*curr_im[i-1,j-1] + -2*curr_im[i-1,j] + -1*curr_im[i-1,j+1]
#           + -2*curr_im[i  ,j-1] + 12*curr_im[i  ,j] + -2*curr_im[i  ,j+1]
#           + -1*curr_im[i+1,j-1] + -2*curr_im[i+1,j] + -1*curr_im[i+1,j+1])

    # Swap references to the images, next_im => curr_im
    #curr_im, next_im = next_im, curr_im
    curr_im = next_im_d.get()
    next_im = curr_im_d.get()

    # Compute the image's pixel mean and variance
    mean, variance = mean_variance(next_im)

  # Save the current image. Clamp the values between 0.0 and 1.0
  img.imsave(out_file_name, curr_im, cmap='gray', vmin=0, vmax=1)
