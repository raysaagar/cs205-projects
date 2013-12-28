import matplotlib.image as img

in_file_name = "Harvard_Small.png"
out_file_name = "Harvard_SimpleRegion_CPU.png"

# Pixel value threshold that we are looking for
threshold = [0, 0.27];

if __name__ == '__main__':
  # Read image. BW images have R=G=B so extract the R-value
  image = img.imread(in_file_name)[:,:,0]
  # Find all pixels within the threshold
  im_region = (threshold[0] <= image) & (image <= threshold[1])
  # Output
  img.imsave(out_file_name, im_region, cmap='gray')
