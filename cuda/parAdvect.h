// CUDA 2D advection solver module
// written for COMP4300/8300 Assignment 2, 2017 
// v1.0 28 Apr

//sets up parallel parameters above
void initParParams(int M, int N, int Gx, int Gy, int Bx, int By, int verb);

// evolve advection on GPU over r timesteps, with (u,ldu) storing the field
// parallel (2D decomposition) variant
void cuda2DAdvect(int r, double *u, int ldu);

// optimized parallel variant
void cudaOptAdvect(int r, double *u, int ldu, int w);
__global__ void updateBoundaryTD(int M, int N, double *u, int ldu);
__global__ void updateBoundaryRL(int N, int M, double *u, int ldu);
__global__ void updateStage1(int M, int N, double *u, int ldu, double *ut,
                              int ldut, double dt, double sx, double sy);
__global__ void updateStage2(int M, int N, double *u, int ldu, double *ut,
                              int ldut, double dtdx, double dtdy);
__global__ void updateOptStage1(int M, int N, double *u, int ldu, double *ut,
                              int ldut, double dt, double sx, double sy);
__global__ void updateOptStage2(int M, int N, double *u, int ldu, double *ut,
                              int ldut, double dtdx, double dtdy);
void loadToBuff1(int x, int y, int i, int j, double *u, double *lb, int ldu, int lblb);
void loadToBuff2(int x, int y, int i, int j, double *ut, double *lb, int ldut, int lblb);
