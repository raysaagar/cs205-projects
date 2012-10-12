from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import math
plt.ion()         # Allow interactive updates to the plots

class data_transformer:
	'''A class to transform a line into a back-projected image'''
	def __init__(self, sample_size, image_size):
		'''Perform the required precomputation for the back-projection step'''
		[self.X,self.Y] = np.meshgrid(np.linspace(-1,1,image_size),
																	np.linspace(-1,1,image_size))
		self.proj_domain = np.linspace(-1,1,sample_size)
		self.f_scale = abs(np.fft.fftshift(np.linspace(-1,1,sample_size+1)[0:-1]))

	def transform(self, data, phi):
		'''Transform a data line taken at an angle phi to its back-projected image'''
		# Compute the Fourier filtered data
		filtered_data = np.fft.ifft(np.fft.fft(data) * self.f_scale).real
		# Interpolate the data to the rotated image domain
		result = np.interp(self.X*np.cos(phi) + self.Y*np.sin(phi),
											 self.proj_domain, filtered_data)
		return result


def parallel_tomo(Transformer, data, comm, n_phi):
  rank = comm.Get_rank()
  size = comm.Get_size()
#  data_block = []
  result = 0
  blocksize = n_phi / size

  if rank == 0:
    for i in xrange(1, size):
      data_block = data[blocksize*i:blocksize*(i+1),:]
      comm.send((data_block, Transformer), dest = i, tag = 15)
    data_block = data[0:blocksize,:]
  else:
    received = comm.recv(source=0, tag=15)
    data_block = received[0]
    Transformer = received[1]

  for k in xrange(0,blocksize):
    # Compute the back-projection
    phi = -(rank * blocksize + k) * math.pi / n_phi
    result += Transformer.transform(data_block[k,:], phi)

  if rank != 0:
    comm.send(result,dest=0, tag=15)
  else:
    for i in xrange(1, size):
      received = comm.recv(source=i, tag=15)
      result += received

  return result

if __name__ == '__main__':
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()

  # Metadata
  n_phi  = 2048        # The number of Tomographic projections
  sample_size = 6144   # The number of samples in each projection

  # process 0 gets all the data
  if rank == 0:
    # Read the projective data from file
    data = np.fromfile(file='TomoData.bin', dtype=np.float64)
    data = data.reshape(n_phi, sample_size)
    # Allocate space for the tomographic image
    ImageSize = 512
    result = np.zeros((ImageSize,ImageSize), dtype=np.float64)
    # Precompute a data_transformer
    Transformer = data_transformer(sample_size, ImageSize)
  else:
    data = None
    Transformer = None

  comm.barrier()
  result = parallel_tomo(Transformer, data, comm, n_phi)
  print np.shape(result)
  comm.barrier()

  # Plot the raw data
  if rank==0:
    plt.figure(1);
    plt.imshow(result, cmap='bone');
    plt.draw();

#  for k in xrange(0,n_phi):
    # Compute the back-projection
#    phi = -k * math.pi / n_phi
#    result += Transformer.transform(data[k,:], phi)

    # Update a plot every so often to show progress
#    print k, phi
#    if k % 50 == 0:
#      plt.figure(2)
#      plt.imshow(result, cmap='bone')
#      plt.draw()

  # Plot/Save the final result
    plt.figure(2)
    plt.imshow(result, cmap=plt.cm.bone)
    plt.draw()
    plt.imsave('TomographicReconstruction4a.png', result, cmap='bone')
  raw_input("Any key to exit...")
