###Background
Stencil computations arise from applying `explicit methods' to the solution of Partial Differential Equations. Most large-scale scientific applications are based on solvers for such equations. A very commonly used component of such solvers will be for the advection process, which models the process of `transport', e.g. wind in an atmosphere simulation. In practice, the most important uses for advection solvers are for 3D phenomena. However, 2D solvers can still model some important problems, e.g. water surface wave propagation, and are considerably simpler to implement. When solved on a 2D regular cartesian grid, advection uses a 9-point stencil (unlike the heat flow problem, which is a 5-point stencil).
The test program simulates the advection (motion) of a sine wave across the unit square. An array u is set to the field values (i.e. water height) across the square accordingly. The process is iterated over a number of timesteps, and the solution will be the field values in u at that point. The boundary conditions are `wrap around', that is field values at x = 0 become those at x = 1 (and conversely). This similarly occurs for the y-dimension. It is possible to compute an exact analytical solution to the advection problem. This can be used to calculate the discretization error in the solution.

The boundary conditions are handled as follows. If the size of the field is MxN, the array u is size (M+2)x(N+2), to store an extra row on the top and bottom, and an extra column to the left and right, to store these boundary values (these are also known as `ghost cells'). That is, the corner elements of the halo are at indices (0,0), (0,N+1), (M+1,N+1), (M+1,0) whereas the corner elements of the interior field are at indices (1,1), (1,N), (M,N), (M,1). In this way, all interior field elements can be updated in a uniform way. In a parallel implementation on a P by Q process grid, halos are also used to store the interior field elements of the neighbouring processes, for the same reason. Unlike the heat flow problem, the corner points for the halos are used in the update of the corner elements of the interior field.

###MPI Setup
project directory will contain a test program advectTest.c, a file serAdvect.c containing a serial advection solver, some header files, and a template parallel advection solver parAdvect.c. The test program can be built using the command make.
It also contains a template ps-ass1Rep.pdf, which you must overwrite with your own report.

The usage for the test program is:

mpirun -np p ./testAdvect [-P P] [-w w] [-o] [-x] M N [r]
with default values p=1, P=1, w=1, r=1. This will run an M by N advection simulation over r timesteps (repetitions) over a P by Q process grid, where p=PQ. w specifies the `halo' width (normally it is 1). If -o is specified, halo communication should be overlapped with computation. The -x is used to invoke an optional extra optimization.
There is also a [-v v] which can be used for debugging (try using -v 1, -v 2 etc).

The test program initializes the local field array (u,ldu), calls the appropriate parallel advection function (in parAdvect.c), and determines the error in the final field. It assumes a 2D block distribution of the global advection field. However, parAdvect.c determines the details of this distribution, and exports:

M_loc, N_loc: the local advection field size (excluding halo).
M0, N0: the local field element (0,0) is global element (M0,N0)
The program computes the `error' in the final solution. As this should not be affected by any parallelization or other optimization of the problem, a comparison of this error with that of the serial case (-np 1) can be used to determine the correctness of any parallelizations.

###OpenMP&CUDA
project directory contains two sub-directories, openmp and cuda. The former contains a test program advectTest.c, a file serAdvect.c containing serial advection functions, some header files, and a template OpenMP advection solver parAdvect.c. The test programs can be built using the command make.
The usage for the test program is:

OMP_NUM_THREADS=p ./testAdvect [-P P] [-x] M N [r]
The test program operates much like that for Assignment 1 except as follows. The -P option invokes an optimization where the parallel region is over all timesteps of the simulation, and P by Q block distribution is used to parallelize the threads, where p=PQ. The -x option is used to invoke an optional extra optimization.
The directory cuda is similar, except the test program is called advectTest.cu, and the template CUDA parallel solver file is called parAdvect.cu. The usage for the test program is:

./testAdvect [-h] [-s] [-g Gx[,Gy]] [-b Bx[,By]] [-o] [-w w] M N [r]
with default values of Gx=Gy=Bx=By=r=1. (Gx,Gy) specifies the grid dimensions of the GPU kernels; (Bx,By) specifies the block dimensions.
The option -h runs the solver on the host; this may be useful for comparing the `error' of the GPU runs (and for comparing GPU and CPU speeds). The option -s forces a serial implementation (run on a single GPU thread); all other options are ignored. If neither of -h,-s,-o are given, Gx,Gy thread blocks of size Bx,By are used in a 2D GPU parallelization of the solver. If -o is specified, an optimized GPU implementation is used, which may use the parameter w as well.