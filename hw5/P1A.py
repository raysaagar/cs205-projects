import numpy as np
import cv2

# PyCUDA modules and device
import pycuda.compiler as nvcc
import pycuda.gpuarray as gpu
import pycuda.autoinit

import time

# Image files
in_file_name = "testimage.jpg"
out_file_name = "P1Aout.png"

# Pixels to remove by seam carving
pix = 1200

def cuda_compile(source_string, function_name):
  # Compile the CUDA Kernel at runtime
  source_module = nvcc.SourceModule(source_string)
  # Return a handle to the compiled CUDA kernel
  return source_module.get_function(function_name)

# w is the original width

# Step 1: energy map kernel source
energy_map_source = \
"""
#define BGR_SUM(pixel) (pixel.x + pixel.y + pixel.z)

__global__ void energy_map(int* e_map, uchar3* img, int Nx, int Ny, int w)
{
    // Calculate center pixel for thread
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int tid = x + y * w;

    if (x < Nx && y < Ny) {
        // Compute energy at center pixel
        int a = BGR_SUM(img[tid]);
	int b;
	int c;
	if (x < Nx - 1)
		b = BGR_SUM(img[tid + 1]);
	else
		b = 0;

	if (y < Ny - 1) 
		c = BGR_SUM(img[tid + w]); 
	else 
		c = 0;
        e_map[tid] = abs(b - a) + abs(c - a);
    }
}
"""

# Step 2: cumulative energy map kernel source
cumulative_energy_map_source = \
"""
__global__ void cumulative_energy_map(int* c_map, int* e_map, int y, int Nx, int Ny, int w)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;

    if (x < Nx) {
        if (y == 0) {
            // Cumulative energy is energy at first row
            c_map[x + y * w] = e_map[x + y * w];
        } else {
            // Compute minimum cumulative energy at pixel
            int next = c_map[x + (y-1) * w];
            if (x > 0)
		next = min(next, c_map[(x-1) + (y-1) * w]);
            if (x < Nx - 1) 
		next = min(next, c_map[(x+1) + (y-1) * w]);
            c_map[x + y * w] = next + e_map[x + y * w];
        }
    }
}
"""

# Stage 3: find start of seam at bottom
min_index_source = \
"""
__global__ void min_index(int* path, int* c_map, int Nx, int Ny, int w)
{
    int min_index = -1;
    int min_val = INT_MAX;

    // Starting index for final row
    int index = (Ny - 1) * w;

    for (int x = 0; x < Nx; x++) {
        // Update minimum value and index if new minimum.
        int newval = c_map[index + x];
        if (newval < min_val) {
            min_val = newval;
            min_index = index + x;
        }
    }

    // Set first element of backtrack path to min_index
    path[Ny - 1] = min_index;
}
"""

# Stage 4:backtrack seam to top of image
backtrack_source = \
"""
__global__ void backtrack(int* path, int* c_map, int Nx, int Ny, int w)
{
    int temp; // fix some pointer ish things 
    for (int y = Ny - 1; y > 0; y--) {
        // Index of center pixel on previous row
        int index = path[y] - w;

        // Index and value of minimum pixel
        int next_pix = index;
        int next_val = c_map[index];

	// x-coord
        int x = index % w;

        // left pixel
        if (x > 0 && next_val > (temp = c_map[index - 1])) {
            next_pix = index - 1;
            next_val = temp;
        }

        // right pixel
        if (x < Nx - 1 && next_val > c_map[index + 1]) {
            next_pix = index + 1;
        }

        path[y - 1] = next_pix;
    }
}
"""

# Stage 5: remove seam from image
remove_seam_source = \
"""
__global__ void remove_seam(uchar3* next, uchar3* current, int* path, int Nx, int Ny, int w)
{
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  int tid = x + y * w;

  int seam = path[y];

  if (x < Nx && y < Ny) {
    if (tid < seam) {
      next[tid] = current[tid];
    }
    else if (tid > seam) {
      next[(x-1) + y * w] = current[tid];
    }
  }
}
"""

def seam_carve(img_to_carve, pix_to_carve):

  # Start Time
  start = time.time()

  # height = Ny, width = Nx
  Ny, Nx, _ = np.int32(img_to_carve.shape)
  size = Nx * Ny

  # CUDA kernels
  kernelstart = time.time()
  energy_map = cuda_compile(energy_map_source, "energy_map")
  cumulative_energy_map = cuda_compile(cumulative_energy_map_source, "cumulative_energy_map")
  min_index = cuda_compile(min_index_source, "min_index")
  backtrack = cuda_compile(backtrack_source, "backtrack")
  remove_seam = cuda_compile(remove_seam_source, "remove_seam")
  totaltime = time.time() - kernelstart
  print "CUDA kernels: %f sec" % totaltime

  # Allocate memory on host and device
  # on host
  img = np.uint8(np.array(img_to_carve))

  # on device; send image buffers
  curr_d = gpu.to_gpu(img) # current image
  next_d = gpu.to_gpu(img) # next image
  e_d = gpu.to_gpu(np.int32(np.zeros((Ny, Nx)))) # energy map
  c_d = gpu.to_gpu(np.int32(np.zeros((Ny, Nx)))) # cumulative energy map
  path_d = gpu.to_gpu(np.int32(np.zeros(Ny))) # seam carving path

  # Original width for indexing
  width = np.int32(Nx)

  # Each iteration removes a seam of 1 pixel
  for i in xrange(pix_to_carve):

    # step 1
    xtpb = 32                               
    ytpb = 16                               
    xblocks = int(np.ceil(Nx * 1.0/xtpb))  
    yblocks = int(np.ceil(Ny * 1.0/ytpb))  
    blocksize = (xtpb, ytpb, 1)
    gridsize  = (xblocks, yblocks)

    # run step 1
    energy_map(e_d, curr_d, Nx, Ny, width, block=blocksize, grid=gridsize)

    # step 2
    tpb = 512
    blocks = int(np.ceil(Nx * 1.0/tpb))
    blocksize = (tpb, 1, 1)
    gridsize  = (blocks, 1)

    # run step 2
    for y in xrange(Ny):
      cumulative_energy_map(c_d, e_d, np.int32(y), Nx, Ny, width, block=blocksize, grid=gridsize)

    # step 3
    blocksize = (1, 1, 1)
    gridsize  = (1, 1)

    # run step 3
    min_index(path_d, c_d, Nx, Ny, width, block=blocksize, grid=gridsize)
    
    # run step 4
    backtrack(path_d, c_d, Nx, Ny, width, block=blocksize, grid=gridsize)

    # step 5
    xtpb = 32                               
    ytpb = 16                               
    xblocks = int(np.ceil(Nx * 1.0/xtpb))  
    yblocks = int(np.ceil(Ny * 1.0/ytpb))  
    blocksize = (xtpb, ytpb, 1)
    gridsize  = (xblocks, yblocks)

    # run step 5
    remove_seam(next_d, curr_d, path_d, Nx, Ny, width, block=blocksize, grid=gridsize)

    # Swap images for next update
    curr_d, next_d = next_d, curr_d

    # Decrease img width by 1 for next update; we removed a seam this round
    Nx = np.int32(Nx - 1)

  # End Time
  totalseamtime = time.time() - start
  print "rescaled frame %dx%d to %dx%d: %f" % (Nx + pix_to_carve, Ny, Nx, Ny, totalseamtime)

  # Return output image
  return curr_d.get()[:,0:Nx,:]

if __name__ == '__main__':
  # Read image
  original_image = cv2.imread(in_file_name)

  # height = Ny, width = Nx
  Ny, Nx, _ = np.int32(original_image.shape)
  print "Processing %d x %d image" % (Nx, Ny)

  # Seam Carve
  output = seam_carve(original_image, pix)

  # Output to file
  cv2.imwrite(out_file_name, output)


