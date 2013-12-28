import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpi4py import MPI
plt.ion()      # Interactive plotting on

class MeshPlotter3D:
  '''A class to help with 3D interactive plotting'''
  def __init__(self, I, J, u):
    '''Perform the required precomputation and make an initial plot'''
    self.fig = plt.figure()
    self.axes = Axes3D(self.fig)
    self.mesh = self.axes.plot_wireframe(I, J, u)
    self.axes.set_zlim3d(-0.25, 0.5)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.draw()

  def draw_now(self, I, J, u):
    '''Update the plot with the data from u'''
    self.mesh.remove()
    self.mesh = self.axes.plot_wireframe(I, J, u)
    self.axes.set_zlim3d(-0.25, 0.5);
    plt.draw();

  def save_now(self, filename):
    self.fig.savefig(filename)


class MeshPlotter3DParallel:
  '''A class to help with 3D interactive plotting from distributed data'''
  def __init__(self, I, J, u, comm=MPI.COMM_WORLD):
    '''Perform the required precomputation and make an initial plot'''
    self.comm = comm
    I, J, u = self.gather_data(I, J, u)
    if self.comm.Get_rank() == 0:
      self.plotter = MeshPlotter3D(I, J, u)

  def gather_data(self, I, J, u):
    # Sanity check
    assert I.size == J.size == u.size

    # Get the size of each distributed portion
    counts = self.comm.gather(u.size, root=0)
    totalsize = np.sum(counts)

    # Allocate a buffer
    if self.comm.Get_rank() == 0:
      I0 = np.zeros(totalsize, dtype=I.dtype)
      J0 = np.zeros(totalsize, dtype=I.dtype)
      u0 = np.zeros(totalsize, dtype=u.dtype)
    else:
      I0, J0, u0 = None, None, None

    # Gather the data with vector-gathers
    self.comm.Gatherv(sendbuf=I.reshape(I.size),
                      recvbuf=(I0, (counts, None)), root=0)
    self.comm.Gatherv(sendbuf=J.reshape(J.size),
                      recvbuf=(J0, (counts, None)), root=0)
    self.comm.Gatherv(sendbuf=u.reshape(u.size),
                      recvbuf=(u0, (counts, None)), root=0)

    # Reorganize
    if self.comm.Get_rank() == 0:
			i0min, j0min = I0.min(), J0.min()
			I0 -= i0min
			J0 -= j0min
			u = np.zeros((I0.max()+1, J0.max()+1))
			I, J = np.zeros(u.shape), np.zeros(u.shape)
			u[I0,J0], I[I0,J0], J[I0,J0] = u0, I0+i0min, J0+j0min
			return I, J, u
    return None, None, None

  def draw_now(self, I, J, u):
    I, J, u = self.gather_data(I, J, u)
    if self.comm.Get_rank() == 0:
      self.plotter.draw_now(I, J, u)

  def save_now(self, filename):
    if self.comm.Get_rank() == 0:
      self.plotter.save_now(filename)
