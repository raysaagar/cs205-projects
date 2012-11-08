import numpy as np
import matplotlib.image as img

# Image files
in_file_name  = "Harvard_Tiny.png"
out_file_name = "Harvard_Sharpened_CPU.png"
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

  # Get image data
  height, width = np.int32(original_image.shape)
  print "Processing %d x %d image" % (width, height)

  # Allocate memory
  curr_im, next_im = np.array(original_image), np.array(original_image)
  # Compute the image's initial mean and variance
  init_mean, init_variance = mean_variance(curr_im)
  variance = init_variance

  while variance < 1.1 * init_variance:
    # Compute Sharpening
    for i in range(1, height-1):
      for j in range(1, width-1):
        next_im[i,j] = curr_im[i,j] + EPSILON * (
             -1*curr_im[i-1,j-1] + -2*curr_im[i-1,j] + -1*curr_im[i-1,j+1]
           + -2*curr_im[i  ,j-1] + 12*curr_im[i  ,j] + -2*curr_im[i  ,j+1]
           + -1*curr_im[i+1,j-1] + -2*curr_im[i+1,j] + -1*curr_im[i+1,j+1])

    # Swap references to the images, next_im => curr_im
    curr_im, next_im = next_im, curr_im

    # Compute the image's pixel mean and variance
    mean, variance = mean_variance(curr_im)

  # Save the current image. Clamp the values between 0.0 and 1.0
  img.imsave(out_file_name, curr_im, cmap='gray', vmin=0, vmax=1)
