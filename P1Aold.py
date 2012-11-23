import numpy as np
import cv2
import time

# PyCUDA modules
import pycuda.compiler as nvcc
import pycuda.gpuarray as gpu
import pycuda.autoinit

# Image files
in_file_name = "testimage.jpg"
out_file_name = "P1Aimg.png"

# number of pixels to seam carve
pix_to_carve = 1200

def cuda_compile(source_str, func_name):
  # Compile the CUDA kernel at runtime
  source_module = nvcc.SourceModule(source_str)
  # Return handle to compiled kernel
  return source_module.get_function(func_name)

# Step 1: energy map kernel
energy_map_source = \
"""
#define BGR(pixel) (pixel.x + pixel.y + pixel.z)

__global__ void energy_map(int* e_map, uchar3* img, int Nx, int Ny, int w)
{
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  int tid = x + y * w;

  // don't want to go over the edge...
  if (x < Nx && y < Ny){
    // compute energy for x,y
    int a = BGR(img[tid]);
    int b, c;
    // compute for x+1,y; ignore if edge
    if(x < Nx - 1)
      b = BGR(img[tid+1]);
    else
      b = 0;
    // compute for x,y+1; ignore if edge
    if(y < Ny - 1)
      c = BGR(img[tid+1]);
    else
      c = 0;
    
    e_map[tid] = abs(b-a) + abs(c-a);
  }
}

"""

# step 2: cumulative energy map kernel
cumulative_energy_map_source= \
"""
__global__ void cumulative_energy_map(int* c_map, int* e_map, int y, int Nx, int Ny, int w)
{
  int x = blockDim.x * blockIdx.x + threadIdx.x;

  if (x < Nx){
    if (y == 0){
      // can't go to y-1, cumulative energy is at first row
      c_map[x+y*w] = e_map[x+y*w];
    }
    else{
      int n = c_map[x+(y-1)*w];
      if (x > 0)
        n = min(n,c_map[(x-1)+(y-1)*w]);
      else if (x < Nx - 1)
        n = min(n, c_map[(x+1)+(y-1)*w]);
      c_map[x+y*w] = n + e_map[x+y*w];
    }
  }
}
"""
# step 3: find seam at bottom row
min_index_source = \
"""
__global__ void min_index(int* path, int* c_map, int Nx, int Ny, int w)
{
  int index = -1; //min index
  int value = INT_MAX; //min value

  // get first elt position
  int r = (Ny - 1) * w;

  for(int x = 0; x < Nx; x++){
    int v = c_map[r + x];

    // update min value if we find a new min elt
    if (v < value){
      value = v;
      index = r + x;
    }
  }
  // set start of seam path
  path[Ny - 1] = index;
}
"""

# step 4: backtrack to find the full seam
backtrack_source = \
"""
__global__ void backtrack(int* path, int* c_map, int Nx, int Ny, int w)
{
  int m;
  for(int y = Ny - 1; y > 0; y--){
    // get previous row (center of x-1/x/x+1)
    int prev_index = path[y] - w;
    
    int next_index = prev_index;
    int value = c_map[prev_index];

    int x = index % w;
    
    if (x > 0 && value > (m = c_map[index - 1])){
      next_index = index - 1;
//      value = c_map[index - 1];
      value = m;
    }

    if (x < Nx - 1 && value > c_map[index - 1]){
      next_index = index + 1;
    }
    // update path
    path[y - 1] = next_index;
  }
}
"""

# step 5: remove seam and shift pixels
seam_remove_source = \
"""
__global__ void seam_remove(uchar3* next, uchar* current, int* path, int Nx, int Ny, int w)
{
  int x = blockDim.x + blockIdx.x + threadIdx.x;
  int y = blockDim.y + blockIdx.y + threadIdx.y;
  int tid = x + y * w;

  int seam = path[y];
  
  if (x < Nx && y < Ny){
    if (tid < seam){
      next[tid] = current[tid];
    }
    else if (tid > seam){
      next[tid - 1] = current[tid];
    }
  }
}
"""
def seam_carve(img, pix_to_carve):
  
  start = time.time()

  height, width, _ = np.int32(img.shape)
  size = width * height

  # compile all the kernels
  compile_start = time.time()
  energy_map = cuda_compile(energy_map_source, "energy_map")
  cumulative_energy_map = cuda_compile(cumulative_energy_map_source, "cumulative_energy_map")
  min_index = cuda_compile(min_index_source, "min_index")
  backtrack = cuda_compile(backtrack_source, "backtrack")
  seam_remove = cuda_compile(seam_remove_source, "seam_remove")
  total_compile = time.time() - compile_start
  print "CUDA kernels: %f" % total_compile

  # allocate space for image

  # on host
  new_img = np.uint8(np.array(img))

  # on device
  currimg_d = gpu.to_gpu(new_img)
  nextimg_d = gpu.to_gpu(new_img)
  e_d = gpu.to_gpu(np.int32(np.zeros((height, width))))
  c_d = gpu.to_gpu(np.int32(np.zeros((height, width))))
  path_d = gpu.to_gpu(np.int32(np.zeros(height)))

  # store original width index
  index = np.int32(width)

  # Every iteration will remove a 1-pixel seam
  for i in xrange(pix_to_carve):
    
    # step 1
    xtpb = 32
    ytbp = 16
    xblocks = int(np.ceil(width * 1.0/xtpb))
    yblocks = int(np.ceil(height * 1.0/ytpb))
    blocksize = (xtpb, ytpb, 1)
    gridsize  = (xblocks, yblocks)

    # run step 1
    energy_map(e_d, currimg_d, width, height, index, block=blocksize, grid=gridsize)

    # step 2
    tpb = 512
    blocks = int(np.ceil(width * 1.0/tpb))
    blocksize = (tpb, 1, 1)
    gridsize = (blocks, 1)

    # run step 2
    for y in xrange(height):
      cumulative_energy_map(c_d, e_d, np.int32(y), width, height, index, block=blocksize, grid=gridsize)

    # step 3
    blocksize = (1,1,1)
    gridsize = (1,1)

    # run step 3
    min_index(path_d, c_d, width, height, index, block=blocksize, grid=gridsize)

    # run step 4
    backtrack(path_d, c_d, width, height, index, block=blocksize, grid=gridsize)

    # step 5
    xtbp = 32
    ytbp = 16
    xblocks = int(np.ceil(width * 1.0/xtpb))
    yblocks = int(np.ceil(height * 1.0/ytpb))
    blocksize = (xtpb, ytpb, 1)
    gridsize  = (xblocks, yblocks)

    # run step 5
    seam_remove(nextimg_d, currimg_d, path_d, width, height, index, block=blocksize, grid=gridsize)

    # update the image
    currimg_d, nextimg_d = nextimg_d, currimg_d

    # width decreased by 1 for next iter
    width = np.int32(width - 1)

  total_time = time.time() - start
  print "Frame rescaled. %dx%d to %dto%d, %f sec" % (width + pix_to_carve, height, width, height, total_time)

  return currimg_d.get()[:,0:width,:]

if __name__ == '__main__':
  # read image
  img = cv2.imread(in_file_name);
  
  height, width, _ = np.int32(img.shape)
  print "Processing %d x %d image" % (width, height)

  # Seam Carve! 
  out_img = seam_carve(img, pix_to_carve)

  # write image
  cv2.imwrite(out_file_name, out_img)
