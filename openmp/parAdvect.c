// OpenMP parallel 2D advection solver module
// written for COMP4300/8300 Assignment 2, 2017
// v1.0 28 Apr

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <omp.h>
#include "serAdvect.h" // advection parameters

static int M, N, P, Q;  //local store of problem parameters
static int verbosity;

//sets up parameters above
void initParParams(int M_, int N_, int P_, int Q_, int verb) {
  M = M_, N = N_; P = P_, Q = Q_;
  verbosity = verb;
} //initParParams()

// uses the Lax-Wendroff method
void myupdateAdvectField(int m, int n, double *u, int ldu) {
  int i, j;
  double sx = 0.5 * Velx / deltax, sy = 0.5 * Vely / deltay;
  double *ut; int ldut = m + 1;
  ut = malloc(ldut * (n + 1) * sizeof(double));
  assert(ut != NULL);
  for (j = 0; j < n + 1; j++)
    for (i = 0; i < m + 1; i++) {
      V(ut, i, j) = 0.25 * (V(u, i, j) + V(u, i - 1, j) + V(u, i, j - 1) + V(u, i - 1, j - 1))
                    - 0.5 * dt * (sy * (V(u, i, j) + V(u, i, j - 1) - V(u, i - 1, j) - V(u, i - 1, j - 1)) +
                                  sx * (V(u, i, j) + V(u, i - 1, j) - V(u, i, j - 1) - V(u, i - 1, j - 1)));
    }
  double dtdx = 0.5 * dt / deltax, dtdy = 0.5 * dt / deltay;
  #pragma omp barrier
  for (j = 0; j < n; j++)
    for (i = 0; i < m; i++) {
      V(u, i, j) +=
        - dtdy * (V(ut, i + 1, j + 1) + V(ut, i + 1, j) - V(ut, i, j) - V(ut, i, j + 1))
        - dtdx * (V(ut, i + 1, j + 1) + V(ut, i, j + 1) - V(ut, i, j) - V(ut, i + 1, j));
    }
  free(ut);
}


// evolve advection over reps timesteps, with (u,ldu) containing the field
// using 1D parallelization
void omp1dAdvect(int reps, double *u, int ldu) {
  int r, i, j;
  int ldut = M + 1;
  double *ut = malloc(ldut * (N + 1) * sizeof(double));

  for (r = 0; r < reps; r++) {
    double sx = 0.5 * Velx / deltax, sy = 0.5 * Vely / deltay;
    double dtdx = 0.5 * dt / deltax, dtdy = 0.5 * dt / deltay;

    #pragma omp parallel for num_threads(P*Q)
    for (i = 1; i < M + 1; i++) { //update left & right boundaries
      V(u, i, 0)   = V(u, i, N);
      V(u, i, N + 1) = V(u, i, 1);
    }

    #pragma omp parallel for num_threads(P*Q)
    for (j = 0; j < N + 2; j++) { //update top & bottom boundaries
      V(u, 0, j)   = V(u, M, j);
      V(u, M + 1, j) = V(u, 1, j);
    }

    u = &V(u, 1, 1); // make u relative to the interior points for the updates

    #pragma omp parallel for num_threads(P*Q)
    for (j = 0; j < N + 1; j++)

      
      // advection update stage 1
       for (i = 0; i < M + 1; i++)
      
        V(ut, i, j) = 0.25 * (V(u, i, j) + V(u, i - 1, j) + V(u, i, j - 1) + V(u, i - 1, j - 1))
                      - 0.5 * dt * (sy * (V(u, i, j) + V(u, i, j - 1) - V(u, i - 1, j) - V(u, i - 1, j - 1)) +
                                    sx * (V(u, i, j) + V(u, i - 1, j) - V(u, i, j - 1) - V(u, i - 1, j - 1)));

    #pragma omp parallel for num_threads(P*Q)
    for(j = 0; j < N; j++)
        
       // advection update stage 2
      for (i = 0; i < M; i++)
       
        V(u, i, j) +=
          - dtdy * (V(ut, i + 1, j + 1) + V(ut, i + 1, j) - V(ut, i, j) - V(ut, i, j + 1))
          - dtdx * (V(ut, i + 1, j + 1) + V(ut, i, j + 1) - V(ut, i, j) - V(ut, i + 1, j));

    u = &V(u, -1, -1); // restore to include the boundary points
  } //for (r...)

  free(ut);
} //omp1dAdvect()


// ... using 2D parallelization
void omp2dAdvect(int reps, double *u, int ldu) {
  int r, i, j;
  
  

  #pragma omp parallel default(shared) private(r, i, j)
  {   
  for (r = 0; r < reps; r++) {
      int tid;
      int P0, Q0;
      int M0, N0;
      int M_loc, N_loc;

      tid = omp_get_thread_num();

      Q0 = tid / P;
      N0 = (N / Q) * Q0;
      N_loc = (Q0 < Q - 1) ? (N / Q) : (N - N0);

      P0 = tid % P;
      M0 = (M / P) * P0;
      M_loc = (P0 < P - 1) ? (M / P) : (M - M0);

      // update up boundary
      if (Q0 == Q - 1)
        for (i = 1 + M_loc * P0; i < 1 + M_loc * (P0 + 1); i++) {
          V(u, i, 0) = V(u, i, N_loc * Q);
          V(u, i, N_loc * Q + 1) = V(u, i, 1);
        }

      #pragma omp barrier

      // update left boundary
      if (P0 == P - 1) {
        for (j = N_loc * Q0; j < 2 + N_loc * (Q0 + 1); j++) {
          V(u, 0, j) = V(u, M_loc * P, j);
          V(u, M_loc * P + 1, j) = V(u, 1, j);
        }
      }


      #pragma omp barrier

      // updatefield
      myupdateAdvectField(M_loc, N_loc, &V(u, M_loc * P0 + 1, N_loc * Q0 + 1), ldu);
      #pragma omp barrier
    }
  }

} //omp2dAdvect()


// ... extra optimization variant
void ompAdvectExtra(int reps, double *u, int ldu) {
  int r, i, j;
  int ldut = M + 1;
  double *ut = malloc(ldut * (N + 1) * sizeof(double));

  for (r = 0; r < reps; r++) {
    double sx = 0.5 * Velx / deltax, sy = 0.5 * Vely / deltay;
    double dtdx = 0.5 * dt / deltax, dtdy = 0.5 * dt / deltay;

    #pragma omp parallel for num_threads(P*Q)
    for (i = 1; i < M + 1; i++) { //update left & right boundaries
      V(u, i, 0)   = V(u, i, N);
      V(u, i, N + 1) = V(u, i, 1);
    }

    #pragma omp parallel for num_threads(P*Q)
    for (j = 0; j < N + 2; j++) { //update top & bottom boundaries
      V(u, 0, j)   = V(u, M, j);
      V(u, M + 1, j) = V(u, 1, j);
    }

    u = &V(u, 1, 1); // make u relative to the interior points for the updates

    
    for (j = 0; j < N + 1; j++)

      #pragma omp parallel for num_threads(P*Q)
      // advection update stage 1
       for (i = 0; i < M + 1; i++)
      
        V(ut, i, j) = 0.25 * (V(u, i, j) + V(u, i - 1, j) + V(u, i, j - 1) + V(u, i - 1, j - 1))
                      - 0.5 * dt * (sy * (V(u, i, j) + V(u, i, j - 1) - V(u, i - 1, j) - V(u, i - 1, j - 1)) +
                                    sx * (V(u, i, j) + V(u, i - 1, j) - V(u, i, j - 1) - V(u, i - 1, j - 1)));


    for (j = 0; j < N; j++)
      #pragma omp parallel for num_threads(P*Q)
       // advection update stage 2
      for(i = 0; i < M; i++)
       
        V(u, i, j) +=
          - dtdy * (V(ut, i + 1, j + 1) + V(ut, i + 1, j) - V(ut, i, j) - V(ut, i, j + 1))
          - dtdx * (V(ut, i + 1, j + 1) + V(ut, i, j + 1) - V(ut, i, j) - V(ut, i + 1, j));

    u = &V(u, -1, -1); // restore to include the boundary points
  } //for (r...)

  free(ut);

} //ompAdvectExtra()
