import numpy as np
import matplotlib.image as img
import time

# Image files
in_file_name  = "Harvard_Small.png"
out_file_name = "Harvard_Sharpened2.png"
# Sharpening constant
EPSILON    = np.float32(.005)

def mean_variance(data):
  '''Return the mean and variance of a 2D array'''
  mean = np.sum(data) / data.size
  variance = np.sum(np.square(data - mean)) / data.size
  print "Mean = %f,  Variance = %f" % (mean, variance)
  return mean, variance

if __name__ == '__main__':
  # Read image. BW images have R=G=B so extract the R-value
  original_image = img.imread(in_file_name)[:,:,0]

  time1 = 0
  time2 = 0
  time3 = 0
  iters = 0

  # Get image data
  height, width = np.int32(original_image.shape)
  h, w = height, width
  i, j = height, width
  print "Processing %d x %d image" % (width, height)

  # Allocate memory
  start = time.time()
  curr_im, next_im = np.array(original_image), np.array(original_image)
  end = time.time()
  time1 = end - start

  # Compute the image's initial mean and variance
  start = time.time()
  init_mean, init_variance = mean_variance(curr_im)
  end = time.time()
  time2 = end - start
  variance = init_variance

  while variance < 1.1 * init_variance:
    # Compute Sharpening
    start = time.time()
    next_im[1:height-1,1:width-1] = curr_im[1:h-1,1:w-1] + EPSILON * (
             -1*curr_im[0:h-2,0:w-2] + -2*curr_im[0:i-2,1:j-1] + 
-1*curr_im[0:i-2,2:j]
           + -2*curr_im[1:i-1,0:j-2] + 12*curr_im[1:i-1,1:j-1] + 
-2*curr_im[1:i-1,2:j]
           + -1*curr_im[2:i,0:j-2] + -2*curr_im[2:i,1:j-1] + 
-1*curr_im[2:i,2:j])
    end = time.time()
    time3 = time3 + end - start

    # Swap references to the images, next_im => curr_im
    curr_im, next_im = next_im, curr_im

    # Compute the image's pixel mean and variance
    start = time.time()
    mean, variance = mean_variance(curr_im)
    end = time.time()
    time2 = time2 + end - start
    iters = iters + 1

  curr_im_fin = curr_im

  # Save the current image. Clamp the values between 0.0 and 1.0
  img.imsave(out_file_name, curr_im_fin, cmap='gray', vmin=0, vmax=1)
  print time1
  print time2
  print time3
  print iters

