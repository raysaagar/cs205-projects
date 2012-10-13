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
  rank = comm.Get_rank()
  # need status updates
  status = MPI.Status()

  while True:
    # receive the job
    i, y = comm.recv(source = 0, tag = MPI.ANY_TAG, status = status)
    
    if status.Get_tag() == 0:
      # if we get the dietag, end the slave process
      return

    C = np.zeros(width, dtype=np.uint16)
    # calculate the mandelrot row
    for j,x in enumerate(np.linspace(minX, maxX,width)):
      C[j] = mandelbrot(x,y)
    # send back the calculated set
    comm.send((i,C), dest=0,tag=rank)

#  return

def master(comm):
  rank = comm.Get_rank()
  size = comm.Get_size()

  image = np.zeros([height,width], dtype=np.uint16)

  status = MPI.Status()
  for i,y in enumerate(np.linspace(minY, maxY, height)):
    # seed the slaves with initial jobs
    if(i < size-1):
      comm.send((i,y),dest=i+1, tag = 1)
    # get result from slave and send next job
    else:
      # if a job is done, receive the row and store it
      row, result = comm.recv(source = MPI.ANY_SOURCE, tag= MPI.ANY_TAG, status=status)
      image[row] = result
      # send new job if there are any left
      comm.send((i,y),dest=status.Get_source(), tag = 1)

  # no more new jobs, collect all outstanding results from slaves
  for n in xrange(1,size):
    row, result = comm.recv(source = MPI.ANY_SOURCE, tag = MPI.ANY_TAG, status = status)
    image[row] = result
    # send dietag to kill slave processes
    comm.send((0,0),dest = status.Get_source(), tag = 0)

  return image


if __name__ == '__main__':
  # Get MPI data
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  # master process will start slaves
  if rank == 0:
    start_time = MPI.Wtime()
    C = master(comm)
    end_time = MPI.Wtime()
    print "Time: %f secs" % (end_time - start_time)
    # comment out if running time tests
    plt.imsave('Mandelbrot.png', C, cmap='spectral')
    plt.imshow(C, aspect='equal', cmap='spectral')
    plt.show()
  else:
    slave(comm)
