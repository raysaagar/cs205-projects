import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt
import math
import pycuda.compiler as nvcc
import pycuda.gpuarray as gpu
import pycuda.driver as cu
import pycuda.autoinit

in_file_name = "Harvard_Small.png"
out_file_name = "fig51.png"

seedThreshold = np.float32(0.08)
threshold = np.float32(0.27)

filter_kernel_source = \
"""
__global__ void filter_kernel(int* inFront, int* outFlags, float* im, int height, int width, float threshold, int frontLen)
{
  int tid = blockIdx.x * blockDim.x * threadIdx.x;

  if (tid < frontLen){
    outFlags[inFront[tid]] = 1;
    if(inFront[tid] % width > 0 && im[inFront[tid] - 1] < threshold)
      outFlags[inFront[tid] - 1] = 1;
    if(inFront[tid] % width < width - 1 && im[inFront[tid] + 1] < threshold)
      outFlags[inFront[tid] + 1] = 1;
    if(inFront[tid] / width > 0 && im[inFront[tid] - width] < threshold)
      outFlags[inFront[tid] - width] = 1;
    if(inFront[tid] / width < height - 1 && im[inFront[tid] + width] < threshold)
      outFlags[inFront[tid] + width] = 1;


  }


}

"""
def cudaCompile(sourceString, functionName): 
  sourceModule = nvcc.SourceModule(sourceString)
  return sourceModule.get_function(functionName)

if __name__ == "__main__":
  original_image = img.imread(in_file_name)[:,:,0]
  height, width = np.int32(original_image.shape)
  print "processing %d x %d image" % (width, height)

  filter_kernel = cudaCompile(filter_kernel_source, "filter_kernel")

  inFrontFlags = np.array(original_image) < seedThreshold
  inFront = np.where(inFrontFlags.ravel() == 1)[0]
  inFrontLen = np.int32(0)
  outFrontLen = np.int32(len(inFront))

  start = cu.Event()
  end = cu.Event()
  
  iter=0
  blocksize = (1024,1,1)

  start.record()
  im_d = gpu.to_gpu(np.float32(np.array(original_image)))
  inFront_d = gpu.to_gpu(np.int32(inFront))
  outFrontFlags_d = gpu.to_gpu(np.int32(np.zeros([height,width])))
  

  while(inFrontLen != outFrontLen):
    inFrontLen = outFrontLen
    filter_kernel(inFront_d, outFrontFlags_d, im_d, height,width, threshold, inFrontLen, block=blocksize, grid=(int(math.ceil(inFrontLen / 1024.0)),1))
    outFrontFlags = outFrontFlags_d.get()
    outFront = np.where(outFrontFlags.ravel() == 1)[0]
    inFront_d = gpu.to_gpu(np.int32(outFront))
    outFrontFlags_d = gpu.to_gpu(np.int32(np.zeros([height, width])))
    outFrontLen = np.int32(len(outFront))
    iter += 1
    print "queue: %d" % outFrontLen

  final_im = np.zeros(height*width)
  final_im[inFront_d.get()] = 1
  final_im = final_im.reshape([height, width])

  end.record()
  end.synchronize()
  totalTime = start.time_till(end) * 1e-3

  img.imsave(out_file_name, final_im, cmap="gray", vmin=0, vmax=1)

  print "run time, %f" % totalTime
  print "iter, %d" % iter


