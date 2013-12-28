import numpy as np
from Plotter3DCS205 import MeshPlotter3D, MeshPlotter3DParallel
import time

# Global constants
xMin, xMax = 0.0, 1.0     # Domain boundaries
yMin, yMax = 0.0, 1.0     # Domain boundaries
Nx = 256#64                   # Numerical grid size
dx = (xMax-xMin)/(Nx-1)   # Grid spacing, Delta x
Ny, dy = Nx, dx           # Ny = Nx, dy = dx
dt = 0.4 * dx             # Time step (Magic factor of 0.4)
T = 7                     # Time end
DTDX = (dt*dt) / (dx*dx)  # Precomputed CFL scalar

# Domain
Gx, Gy = Nx+2, Ny+2     # Numerical grid size with ghost cells
Ix, Iy = Nx+1, Ny+1     # Convience so u[1:Ix,1:Iy] are all interior points

def initial_conditions(x0, y0):
  '''Construct the grid cells and set the initial conditions'''
  um = np.zeros([Gx,Gy])     # u^{n-1}  "u minus"
  u  = np.zeros([Gx,Gy])     # u^{n}    "u"
  up = np.zeros([Gx,Gy])     # u^{n+1}  "u plus"
  # Set the initial condition on interior cells: Gaussian at (x0,y0)
  [I,J] = np.mgrid[1:Ix, 1:Iy]
  u[1:Ix,1:Iy] = np.exp(-50 * (((I-1)*dx-x0)**2 + ((J-1)*dy-y0)**2))
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
  up[1:Ix,1:Iy] = ((2-4*DTDX)*u[1:Ix,1:Iy] - um[1:Ix,1:Iy]
                   + DTDX*(u[0:Ix-1,1:Iy] + u[2:Ix+1,1:Iy] +
                           u[1:Ix,0:Iy-1] + u[1:Ix,2:Iy+1]))

  # The above is a vectorized operation for the simple for-loops:
  #for i in range(1,Ix):
  #  for j in range(1,Iy):
  #    up[i,j] = ((2-4*DTDX)*u[i,j] - um[i,j]
  #               + DTDX*(u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1]))

def set_ghost_cells(u):
  '''Set the ghost cells. In serial, the only ghost cells are the boundaries.
  In parallel, each process will have ghost cells: some will need data from
  neighboring processes, others will use these boundary conditions.'''
  # Update Ghost Cells with Boundary Condition
  u[0,:]    = u[2,:];       # u_{0,j}    = u_{2,j}
  u[Nx+1,:] = u[Nx-1,:];    # u_{Nx+1,j} = u_{Nx-1,j}
  u[:,0]    = u[:,2];       # u_{i,0}    = u_{i,2}
  u[:,Ny+1] = u[:,Ny-1];    # u_{i,Ny+1} = u_{i,Ny-1}


if __name__ == '__main__':
  # Set the initial conditions
  up, u, um = initial_conditions(0.5, 0)

  # Setup and draw the first frame with the interior points
  [I,J] = np.mgrid[1:Ix,1:Iy]   # The global indices for u[1:Ix,1:Iy]
#  plotter = MeshPlotter3D(I, J, u[1:Ix,1:Iy])

  s_start = time.time()

  for k,t in enumerate(np.arange(0,T,dt)):
    # Compute u^{n+1} with the computational stencil
    apply_stencil(up, u, um)

    # Swap references for the next step
    # u^{n-1} <- u^{n} and u^{n} <- u^{n+1} and u^{n+1} garbage
    um, u, up = u, up, um

    # Set the ghost cells on u
    set_ghost_cells(u)

    # Output and draw Occasionally
 #   print "Step: %d  Time: %f" % (k,t)
 #   if k % 5 == 0:
 #     plotter.draw_now(I, J, u[1:Ix,1:Iy])

  s_stop = time.time()
  total = s_stop - s_start
  avgT = total/(T/dt)
  print "Time: %f" % total
  print "Time/iteration: %f" % avgT

  plotter = MeshPlotter3D(I, J, u[1:Ix,1:Iy])
  plotter.save_now("FinalWave.png")
