// OpenMP parallel 2D advection solver module
// written for COMP4300/8300 Assignment 2, 2017
// v1.0 28 Apr

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "serAdvect.h" // advection parameters
#include "parAdvect.h"

static int M, N, Gx, Gy, Bx, By; // local store of problem parameters
static int verbosity;

//sets up parameters above
void initParParams(int M_, int N_, int Gx_, int Gy_, int Bx_, int By_,
                   int verb) {
  M = M_, N = N_; Gx = Gx_; Gy = Gy_;  Bx = Bx_; By = By_;
  verbosity = verb;
} //initParParams()


// evolve advection over reps timesteps, with (u,ldu) containing the field
// parallel (2D decomposition) variant
void cuda2DAdvect(int reps, double *u, int ldu) {

  dim3 threadsPerBlock(Bx, By);
  dim3 numBlock(Gx, Gy);
  int ldut = M + 1;
  double *ut;
  HANDLE_ERROR( cudaMalloc(&ut, ldut * (N + 1)*sizeof(double)) );
  for (int r = 0; r < reps; r++) {


    updateBoundaryTD <<< Gx*Gy, Bx*By >>> (M, N, u, ldu);
    updateBoundaryRL <<< Gx*Gy, Bx*By >>> (N, M, u, ldu);

    double sx = 0.5 * Velx / deltax, sy = 0.5 * Vely / deltay;
    updateStage1 <<< numBlock, threadsPerBlock>>> (M, N, &V(u, 1, 1), ldu, ut, ldut, dt, sx, sy);

    double dtdx = 0.5 * dt / deltax, dtdy = 0.5 * dt / deltay;
    updateStage2 <<< numBlock, threadsPerBlock>>> (M, N, &V(u, 1, 1), ldu, ut, ldut, dtdx, dtdy);



  } //for(r...)
  HANDLE_ERROR( cudaFree(ut) );

} //cuda2DAdvect()



// ... optimized parallel variant
void cudaOptAdvect(int reps, double *u, int ldu, int w) {

  dim3 threadsPerBlock(Bx, By);
  dim3 numBlock(Gx, Gy);
  int ldut = M + 1;
  double *ut;
  int n = (Bx + 1) * (By + 1);
  HANDLE_ERROR( cudaMalloc(&ut, ldut * (N + 1)*sizeof(double)) );
  for (int r = 0; r < reps; r++) {

    updateBoundaryTD <<< numBlock, threadsPerBlock >>> (M, N, u, ldu);
    updateBoundaryRL <<< numBlock, threadsPerBlock >>> (N, M, u, ldu);

    double sx = 0.5 * Velx / deltax, sy = 0.5 * Vely / deltay;
    updateOptStage1 <<< numBlock, threadsPerBlock, n * sizeof(double)>>> (M, N, &V(u, 1, 1), ldu, ut, ldut, dt, sx, sy);

    double dtdx = 0.5 * dt / deltax, dtdy = 0.5 * dt / deltay;
    updateOptStage2 <<< numBlock, threadsPerBlock, n * sizeof(double)>>> (M, N, &V(u, 1, 1), ldu, ut, ldut, dtdx, dtdy);



  } //for(r...)
  HANDLE_ERROR( cudaFree(ut) );

} //cudaOptAdvect()

/********************** Cuda Kernels **********************************/

__global__ void updateStage1(int M, int N, double *u, int ldu, double *ut,
                             int ldut, double dt, double sx, double sy) {
  int i  = blockIdx.x * blockDim.x + threadIdx.x;
  int j  = blockIdx.y * blockDim.y + threadIdx.y;
  int start = j;

  while (i < M + 1) {
    j = start;
    while (j < N + 1) {

      V(ut, i, j) = 0.25 * (V(u, i, j) + V(u, i - 1, j) + V(u, i, j - 1) + V(u, i - 1, j - 1))
                    - 0.5 * dt * (sy * (V(u, i, j) + V(u, i, j - 1) - V(u, i - 1, j) - V(u, i - 1, j - 1)) +
                                  sx * (V(u, i, j) + V(u, i - 1, j) - V(u, i, j - 1) - V(u, i - 1, j - 1)));

      j += blockDim.y * gridDim.y;
    }
    i += blockDim.x * gridDim.x;
  }
}


__global__ void updateStage2(int M, int N, double *u, int ldu, double *ut,
                             int ldut, double dtdx, double dtdy) {
  int i  = blockIdx.x * blockDim.x + threadIdx.x;
  int j  = blockIdx.y * blockDim.y + threadIdx.y;
  int start = j;
  while (i < M) {
    j = start;
    while (j < N) {
      V(u, i, j) +=
        - dtdy * (V(ut, i + 1, j + 1) + V(ut, i + 1, j) - V(ut, i, j) - V(ut, i, j + 1))
        - dtdx * (V(ut, i + 1, j + 1) + V(ut, i, j + 1) - V(ut, i, j) - V(ut, i + 1, j));

      j += blockDim.y * gridDim.y;
    }
    i += blockDim.x * gridDim.x;
  }
}



__global__ void updateBoundaryTD(int M, int N, double *u, int ldu) {
  int i =  blockIdx.x * blockDim.x + threadIdx.x;        
  while ( i < M ) {
    V(u, i + 1, 0)   = V(u, i + 1, N);
    V(u, i + 1, N + 1) = V(u, i + 1, 1);
    i += blockDim.x * gridDim.x;
  }
}
__global__ void updateBoundaryRL(int N, int M, double *u, int ldu) {
  int j =  blockIdx.x * blockDim.x + threadIdx.x;
  while (j < N + 2) {
    V(u, 0, j)   = V(u, M, j);
    V(u, M + 1, j) = V(u, 1, j);
    j += blockDim.x * gridDim.x;
  }
}

/********************** Opt Cuda Kernels **********************************/

__global__ void updateOptStage1(int M, int N, double *u, int ldu, double *ut,
                                int ldut, double dt, double sx, double sy) {
  extern __shared__ double localBuff[];
  int ldlb = blockDim.x + 1;
  int ldlocalBuff =  blockDim.x + 1;
  int i  = blockIdx.x * blockDim.x + threadIdx.x;
  int j  = blockIdx.y * blockDim.y + threadIdx.y;
  int x = threadIdx.x;
  int y = threadIdx.y;
  int start = j;

  double *lb = &V(localBuff, 1, 1);

  while (i < M + 1) {
    j = start;
    while (j < N + 1) {

      V(lb, x, y) = V(u, i, j);
      if ( x == 0) {
        V(lb, x-1, y) = V(u, i-1, j);
      }
      if ( y == 0) {
        V(lb, x, y-1) = V(u, i, j-1);
      }
      if ( x == 0 && y == 0) {
        V(lb, x-1, y-1) = V(u, i-1, j-1);
      }
      __syncthreads();
      V(ut, i, j) = 0.25 * (V(lb, x, y) + V(lb, x - 1, y) + V(lb, x, y - 1) + V(lb, x - 1, y - 1))
                    - 0.5 * dt * (sy * (V(lb, x, y) + V(lb, x, y - 1) - V(lb, x - 1, y) - V(lb, x - 1, y - 1)) +
                                  sx * (V(lb, x, y) + V(lb, x - 1, y) - V(lb, x, y - 1) - V(lb, x - 1, y - 1)));
      // }
      __syncthreads();
      j += blockDim.y * gridDim.y;
    }
    __syncthreads();
    i += blockDim.x * gridDim.x;
  }
}
__global__ void updateOptStage2(int M, int N, double *u, int ldu, double *ut,
                                int ldut, double dtdx, double dtdy) {
  int ldlb = blockDim.x + 1;
  extern __shared__ double lb[];
  int i  = blockIdx.x * blockDim.x + threadIdx.x;
  int j  = blockIdx.y * blockDim.y + threadIdx.y;
  int x = threadIdx.x;
  int y = threadIdx.y;
  int start = j;

  while (i < M) {
    j = start;
    while (j < N) {

      V(lb, x, y) = V(ut, i, j);
      if ( x == blockDim.x - 1) {
        V(lb, x + 1, y) = V(ut, i + 1, j);
      }
      if ( y == blockDim.y - 1) {
        V(lb, x, y + 1) = V(ut, i, j + 1);
      }
      if ( x == blockDim.x - 1 && y == blockDim.y - 1) {
        V(lb, x + 1, y + 1) = V(ut, i + 1, j + 1);
      }
      __syncthreads();

      V(u, i, j) +=
        - dtdy * (V(lb, x + 1, y + 1) + V(lb, x + 1, y) - V(lb, x, y) - V(lb, x, y + 1))
        - dtdx * (V(lb, x + 1, y + 1) + V(lb, x, y + 1) - V(lb, x, y) - V(lb, x + 1, y));
      __syncthreads();
      j += blockDim.y * gridDim.y;
    }
    __syncthreads();
    i += blockDim.x * gridDim.x;
  }
}



