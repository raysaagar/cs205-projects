import numpy as np
import matplotlib.pyplot as plt
import time
from mpi4py import MPI

def mandelbrot(x, y):
  '''Compute a Mandelbrot pixel'''
  z = c = complex(x,y)
  it, maxit = 0, 511
  while abs(z) < 2 and it < maxit:
    z = z*z + c
    it += 1
  return it

# Global variables, can be used by any process
minX,  maxX   = -2.1, 0.7
minY,  maxY   = -1.25, 1.25
width, height = 2**10, 2**10

# build mandelbrot block
def parallel_mandelbrot(comm):
  rank = comm.Get_rank()
  size = comm.Get_size()
  # calculate blocksize
  blocksize = int(np.ceil(height/float(size)))
  # store y values for mandelbrot calculation
  ylist = np.linspace(minY, maxY, height)
  # empty buffer
  C = np.zeros([blocksize,width], dtype=np.uint16)

  # loop over all the rows to calculate the mandelbrot
  for i in xrange(0,blocksize):
    for j,x in enumerate(np.linspace(minX, maxX, width)):
      if rank*blocksize+i >= height:
        break # we don't want to go out of bounds!
      row = rank*blocksize+i
      # store calculated value
      C[i,j] = mandelbrot(x,ylist[row])

  return C


if __name__ == '__main__':
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  size = comm.Get_size()

  # start the timer
  comm.barrier()
  start = MPI.Wtime()

  # initial collection
  C = parallel_mandelbrot(comm)

  # rank 0 will control the run and final results
  if rank == 0:
    # blocksize 
    n = int(np.ceil(float(height)/size))
    mandel = np.zeros([n*size,width],dtype=np.uint16)
    # use gather to collect all results from processes
    comm.Gather(C, mandel, root=0)
    comm.barrier()
    end = MPI.Wtime()
    print "Total time: %f" % (end - start)
    plt.imsave('Mandlebrot_joe.png', C[0:height,:], cmap='spectral')
  else:
    comm.Gather(C, None, root=0)
    comm.barrier()
