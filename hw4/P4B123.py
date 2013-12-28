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

# Define the CUDA kernel as a string.
sharpen_kernel_source = \
"""
__global__ void sharpen_kernel(float* curr_im, float* next_im, int height, int width, float epsilon)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int gid = width * y + x;

  if (y > 0 && x > 0 && y < height-1 && x < width-1){
    next_im[gid] = curr_im[gid] + epsilon * (
      -1*curr_im[gid-width-1] + -2*curr_im[gid-1] + -1*curr_im[gid+width-1]
      + -2*curr_im[gid-width] + 12*curr_im[gid] + -2*curr_im[gid+width]
      + -1*curr_im[gid-width+1] + -2*curr_im[gid+1] + -1*curr_im[gid+width+1]);
  }
}
"""

# Image files
in_file_name  = "Harvard_Small.png"
out_file_name = "Harvard_Sharpened_CUDA_CPU_SmallB.png"
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
  
  #initstart = time.time()
  # Allocate device memory and copy host to device
  curr_im_d = gpu.to_gpu(curr_im)
  next_im_d = gpu.to_gpu(next_im)
  #initTime = time.time() - initstart

  # ReductionKernel
  mean_kernel = ReductionKernel(dtype_out=np.float32, neutral="0",
          reduce_expr="a+b", map_expr="x[i]/(height*width)",
                  arguments="float* x, int height, int width")
  
  var_kernel = ReductionKernel(dtype_out=np.float32, neutral="0",
          reduce_expr="a+b", map_expr="((x[i]-mean)*(x[i]-mean))/(height*width)",
                  arguments="float* x, int height, int width, float mean")


  while variance < 1.1 * init_variance:

    sharpen_kernel(curr_im_d, next_im_d, height,width,EPSILON, 
      block = (TPBx, TPBy, 1), grid=(nBx, nBy))


    mean = mean_kernel(next_im_d, height, width).get()
    variance = var_kernel(next_im_d, height, width, mean).get()
    print "Mean = %f,  Variance = %f" % (mean, variance)
    #x_dot_y_cpu = numpy.dot(x_gpu.get(), y_gpu.get())
    
    #swap refs
    curr_im_d, next_im_d = next_im_d, curr_im_d

    # Compute the image's pixel mean and variance
    #mean, variance = mean_variance(curr_im)

  # Swap references to the images, next_im => curr_im
  curr_im = next_im_d.get()
  next_im = curr_im_d.get()

  # Save the current image. Clamp the values between 0.0 and 1.0
  img.imsave(out_file_name, curr_im, cmap='gray', vmin=0, vmax=1)
