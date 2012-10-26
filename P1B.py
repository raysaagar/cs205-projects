import numpy as np
from Plotter3DCS205 import MeshPlotter3D, MeshPlotter3DParallel
from mpi4py import MPI

# Global constants
xMin, xMax = 0.0, 1.0     # Domain boundaries
yMin, yMax = 0.0, 1.0     # Domain boundaries
Nx = 64                   # Numerical grid size
dx = (xMax-xMin)/(Nx-1)   # Grid spacing, Delta x
Ny, dy = Nx, dx           # Ny = Nx, dy = dx
dt = 0.4 * dx             # Time step (Magic factor of 0.4)
T = 7                     # Time end
DTDX = (dt*dt) / (dx*dx)  # Precomputed CFL scalar
Px = 4
Py = 2

# Domain
Gx, Gy = Nx+2, Ny+2     # Numerical grid size with ghost cells
Ix, Iy = Nx+1, Ny+1     # Convience so u[1:Ix,1:Iy] are all interior points

def initial_conditions(x0, y0, row, col):
  '''Construct the grid cells and set the initial conditions'''
#  um = np.zeros([Gx,Gy])     # u^{n-1}  "u minus"
#  u  = np.zeros([Gx,Gy])     # u^{n}    "u"
#  up = np.zeros([Gx,Gy])     # u^{n+1}  "u plus"
  um = np.zeros([Nx/Px+2,Ny/Py+2])
  u = np.zeros([Nx/Px+2,Ny/Py+2])
  up = np.zeros([Nx/Px+2, Ny/Py+2])

  # Set the initial condition on interior cells: Gaussian at (x0,y0)
  [I,J] = np.mgrid[globx:globx+(Nx/Px), globy:globy+(Ny/Py)]
  u[1:LIx,1:LIy] = np.exp(-50 * (((I-1)*dx-x0)**2 + ((J-1)*dy-y0)**2))
  # Set the ghost cells to the boundary conditions
  set_ghost_cells(u)
  # Set the initial time derivative to zero by running backwards
  apply_stencil(um, u, up)
  um *= 0.5
  # Done initializing up, u, and um
  return up, u, um

def apply_stencil(up, u, um):
  '''Apply the computational stencil to compute up.
  Assumes the ghost cells exist and are set to the correct values.'''
  # Update interior grid cells with vectorized stencil
  up[1:LIx,1:LIy] = ((2-4*DTDX)*u[1:LIx,1:LIy] - um[1:LIx,1:LIy]
                   + DTDX*(u[0:LIx-1,1:LIy] + u[2:LIx+1,1:LIy] +
                           u[1:LIx,0:LIy-1] + u[1:LIx,2:LIy+1]))

  # The above is a vectorized operation for the simple for-loops:
  #for i in range(1,LIx):
  #  for j in range(1,LIy):
  #    up[i,j] = ((2-4*DTDX)*u[i,j] - um[i,j]
  #               + DTDX*(u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1]))

def set_ghost_cells(u):
  '''Set the ghost cells. In serial, the only ghost cells are the boundaries.
  In parallel, each process will have ghost cells: some will need data from
  neighboring processes, others will use these boundary conditions.'''
  coords = cart_comm.Get_coords(rank)

  #top row
  if (coords[0] == 0):
    u[0,:] = u[2,:]
  else:
    b_in = np.zeros([1,LGy])
    neighbor = cart_comm.Get_cart_rank([coords[0]-1,coords[1]])
    
    req = cart_comm.Irecv(b_in[0,:],source=neighbor,tag=0)
    cart_comm.Isend(u[1,:],dest=neighbor,tag=0)

    status = MPI.Status()
    req.Wait(status)
    assert status.source == neighbor
#    assert status.tag == 0

    u[0,:] = b_in[0,:]

  if (coords[0] == (Px-1)):
    u[Lx+1,:] = u[Lx-1,:]
  else:
    b_in = np.zeros([1,LGy])
    neighbor = cart_comm.Get_cart_rank([coords[0]+1,coords[1]])

    req = cart_comm.Irecv(b_in[0,:],source=neighbor,tag=0)
    cart_comm.Isend(u[Lx,:],dest=neighbor,tag=0)

    status = MPI.Status()
    req.Wait(status)
    assert status.source == neighbor
#    assert status.tag == 0

    u[Lx+1,:] = b_in[0,:]

  #left col
  if (coords[1] == 0):
    u[:,0] = u[:,2]
  else:
    b_in, b_out = np.zeros([1,LGx]), np.zeros([1,LGx])
    b_out[0,:] = np.transpose(u[:,1])
    neighbor = cart_comm.Get_cart_rank([coords[0],coords[1]-1])
    
    req = cart_comm.Irecv(b_in[0,:], source = neighbor, tag = 1)
    cart_comm.Isend(b_out[0,:],dest=neighbor, tag=1)

    status = MPI.Status()
    req.Wait(status)
    assert status.source == neighbor
#    assert status.tag == 1
    
    u[:,0] = np.transpose(b_in[0,:])
  
  #right col
  if(coords[1] == (Py-1)):
    u[:,Ly+1] = u[:,Ly-1]
  else:
    b_in, b_out = np.zeros([1,LGx]), np.zeros([1,LGx])
    b_out[0,:] = np.transpose(u[:,Ly])
    neighbor = cart_comm.Get_cart_rank([coords[0],coords[1]+1])
    
    req = cart_comm.Irecv(b_in[0,:], source=neighbor, tag=1)
    cart_comm.Isend(b_out[0,:],dest=neighbor,tag=1)

    status = MPI.Status()
    req.Wait(status)
    assert status.source == neighbor
#    assert status.tag == 1
    
    u[:,Ly+1] = np.transpose(b_in[0,:])

  # Update Ghost Cells with Boundary Condition
  #u[0,:]    = u[2,:];       # u_{0,j}    = u_{2,j}
  #u[Nx/Px+1,:] = u[Nx/Px-1,:];    # u_{Nx+1,j} = u_{Nx-1,j}
  #u[:,0]    = u[:,2];       # u_{i,0}    = u_{i,2}
  #u[:,Ny/Py+1] = u[:,Ny/Py-1];    # u_{i,Ny+1} = u_{i,Ny-1}


if __name__ == '__main__':
  
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  size = comm.Get_size()

  cart_comm = comm.Create_cart((Px,Py))
  coords = cart_comm.Get_coords(rank)
  row,col = coords[0],coords[1]

  #global values
  global Lx, Ly, LGx, LGy, LIx, LIy
  Lx, Ly = Nx/Px, Ny/Py
  LGx, LGy = Lx+2, Ly+2
  LIx, LIy = Lx+1, Ly+1

  #global index
  globx, globy = Lx*row,Ly*col

  # Set the initial conditions
  up, u, um = initial_conditions(0.5, 0, globx, globy)

  # Setup and draw the first frame with the interior points
#  [I,J] = np.mgrid[1:Ix,1:Iy]   # The global indices for u[1:Ix,1:Iy]
  [I,J] = np.mgrid[globx:globx+(Nx/Px), globy:globy+(Ny/Py)]
  #plotter = MeshPlotter3DParallel(I, J, u[1:LIx,1:LIy])

  comm.barrier()
  p_start = MPI.Wtime()

  for k,t in enumerate(np.arange(0,T,dt)):
    # Compute u^{n+1} with the computational stencil
    apply_stencil(up, u, um)

    # Swap references for the next step
    # u^{n-1} <- u^{n} and u^{n} <- u^{n+1} and u^{n+1} garbage
    um, u, up = u, up, um

    # Set the ghost cells on u
    set_ghost_cells(u)

    # Output and draw Occasionally
    #if(rank == 0):
    #  print "Step: %d  Time: %f" % (k,t)
    #if k % 5 == 0:
      #plotter.draw_now(I, J, u[1:LIx,1:LIy])

  comm.barrier()
  time = (MPI.Wtime() - p_start)
  avgT = time / (T/dt)

  if (rank == 0):
    print "Time: %f" % time
    print "Time/iteration: %f" % avgT

  plotter = MeshPlotter3DParallel(I, J, u[1:LIx,1:LIy])
  if(rank == 0):
    plotter.save_now("FinalWaveA.png")
