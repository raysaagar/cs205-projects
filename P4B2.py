import numpy as np
import matplotlib.image as img

# Import the PyCUDA modules
import pycuda.compiler as nvcc
import pycuda.gpuarray as gpu
import pycuda.driver as cu
from pycuda.reduction import ReductionKernel
from pycuda.elementwise import ElementwiseKernel 

# Initialize the CUDA device
import pycuda.autoinit

# Image files
in_file_name  = "Harvard_Medium.png"
out_file_name = "fig41.png"

# Sharpening constant
EPSILON    = np.float32(.005)

sharpen_kernel_source = \
"""
__global__ void sharpen_kernel(float* curr_im, float* next_im, int height, int width, float epsilon)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int tid = width*y + x;
  
  if (y > 0 && y < height - 1 && x > 0 && x < width - 1){
    next_im[tid] = curr_im[tid] + epsilon * 
        (-1*curr_im[tid-width-1] + -2*curr_im[tid-width] + -1*curr_im[tid-width+1]
        + -2*curr_im[tid-1] + 12*curr_im[tid] + -2 * curr_im[tid+1]
        + -1*curr_im[tid+width-1] + -2 * curr_im[tid + width] + -1 * curr_im[tid + width + 1]);
  }

}

"""


def meanVariance(data):
  mean = np.sum(data) / data.size
  variance = np.sum(np.square(data - mean)) / data.size
  print "mean = %f, variance = %f" % (mean, variance)
  return mean, variance

def cudaCompile(sourceString, functionName): 
  sourceModule = nvcc.SourceModule(sourceString)
  return sourceModule.get_function(functionName)

if __name__ == "__main__":
  original_image = img.imread(in_file_name)[:,:,0]
  height,width = np.int32(original_image.shape)
  print "processing %d x %d image" % (width, height)

  curr, next = np.array(original_image), np.array(original_image)
  initialMean, initialVar = meanVariance(curr)
  currentVar = initialVar

  sharpen_kernel = cudaCompile(sharpen_kernel_source, "sharpen_kernel")

  TPBx = 32
  TPBy = 32
  nBx = int(width/TPBx)
  nBy = int(height/TPBy)

  meanKernel = ReductionKernel(dtype_out = np.float32, neutral = "0", reduce_expr = "a + b", map_expr = "x[i] / (height * width)", arguments = "float* x, int height, int width")

  varianceKernel = ReductionKernel(dtype_out = np.float32, neutral = "0", reduce_expr = "a + b", map_expr = "((x[i] - mean) * (x[i] - mean)) / (height * width)", arguments = "float* x, int height, int width, float mean")

  initStart = cu.Event()
  initEnd = cu.Event()
  initStart.record()
  curr_d = gpu.to_gpu(curr)
  next_d = gpu.to_gpu(next)
  initEnd.record()
  initEnd.synchronize()
  initTime = initStart.time_till(initEnd) * 1e-3

  iter = 0
  sharpenTime = 0
  mvTime = 0

  while currentVar < 1.1 * initialVar:
    sharpenStart = cu.Event()
    sharpenEnd = cu.Event()
    sharpenStart.record()
    sharpen_kernel(curr_d, next_d, height, width, EPSILON, block = (TPBx, TPBy, 1), grid = (nBx,nBy))
    sharpenEnd.record()
    sharpenEnd.synchronize()
    sharpenTime += sharpenStart.time_till(sharpenEnd) * 1e-3

    mvStart = cu.Event()
    mvEnd = cu.Event()
    mvStart.record()
    mean = meanKernel(next_d, height, width).get()
    currentVar = varianceKernel(next_d, height, width, mean).get()
    mvEnd.record()
    mvEnd.synchronize()
    mvTime += mvStart.time_till(mvEnd) * 1e-3

    print "mean = %f, variance = %f" % (mean, currentVar)
    
    curr_d, next_d = next_d, curr_d
    iter += 1


  curr = curr_d.get()
  img.imsave(out_file_name, curr, cmap = "gray", vmin = 0, vmax = 1)


  print "initialization time, %f" % initTime
  print "sharpening time, %f " % (sharpenTime)
  print "mean/variance time, %f" % (mvTime)
  print "iter, %f" % iter






