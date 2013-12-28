import numpy as np
import matplotlib.image as img
import time

in_file_name = "Harvard_Small.png"
out_file_name = "Harvard_GrowRegion_CPU.png"

# Region growing constants [min, max]
seed_threshold = [0, 0.08];
threshold      = [0, 0.27];

def in_range(value, pair):
  '''True if value is in the range [pair[0],pair[1]]'''
  return pair[0] <= value <= pair[1]

def add_neighbors2D(i, j, width, height, data):
  '''Add all valid neighbor indices of (i,j) to the set data'''
  index = i*width + j        # Index of (i,j)
  if i > 0:
    data.add(index - width)  # Add (i-1,j)
  if i < height-1:
    data.add(index + width)  # Add (i+1,j)
  if j > 0:
    data.add(index - 1)      # Add (i,j-1)
  if j < width-1:
    data.add(index + 1)      # Add (i,j+1)

def index2ij(index, width, height):
  '''Transform an index to an (i,j) pair'''
  return index / width, index % width


if __name__ == '__main__':
  # Read image. BW images have R=G=B so extract the R-value
  image = img.imread(in_file_name)[:,:,0]
  height, width = np.int32(image.shape)
  print "Processing %d x %d image" % (width, height)

  # Initialize the image region as empty
  im_region = np.int8(np.zeros([height, width]))

  start_time = time.time()

  # Find any seed points in the image and add them to the region
  next_front = set()
  for i in xrange(0, height):
    for j in xrange(0, width):
      if in_range(image[i,j], seed_threshold):
        # This pixel is a seed pixel! Add it to the image region
        im_region[i,j] = 1
        # Add its neighbours to the next front
        add_neighbors2D(i, j, width, height, next_front)

  # While the next front has pixels we need to process
  while len(next_front) > 0:
    # Swap front sets, this_front <= next_front, next_front = new set
    this_front, next_front = next_front, set()
    # For all the indices in this front
    for index in this_front:
      i,j = index2ij(index, width, height)  # Get (i,j)
      # If this front pixel isn't in the region, but it belongs in it
      if im_region[i,j] == 0 and in_range(image[i,j], threshold):
        # Add it!
        im_region[i,j] = 1;
        # Add its neighbours to the next front
        add_neighbors2D(i, j, width, height, next_front)

  # Output
  stop_time = time.time()
  print "Serial: %f" % (stop_time - start_time)
  img.imsave(out_file_name, im_region, cmap='gray')

