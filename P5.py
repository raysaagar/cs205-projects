import numpy as np
import matplotlib.pyplot as plt
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

def slave(comm):
  return

def master(comm):
  image = np.zeros([height,width], dtype=np.uint16)
  return image


if __name__ == '__main__':
  # Get MPI data
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()

  if rank == 0:
    start_time = MPI.Wtime()
    C = master(comm)
    end_time = MPI.Wtime()
    print "Time: %f secs" % (end_time - start_time)
    plt.imsave('Mandelbrot.png', C, cmap='spectral')
    #plt.imshow(C, aspect='equal', cmap='spectral')
    #plt.show()
  else:
    slave(comm)
