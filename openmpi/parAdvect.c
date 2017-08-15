// parallel 2D advection solver module
// written for COMP4300/8300 Assignment 1 
// v1.0 15 Mar 

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <assert.h>
#include <string.h>
#include "serAdvect.h"

#define HALO_TAG 100

int M_loc, N_loc; // local advection field size (excluding halo) 
int M0, N0;       // local field element (0,0) is global element (M0,N0)
static int P0, Q0; // 2D process id (P0, Q0) in P x Q process grid 

static int M, N, P, Q; // local store of problem parameters
static int verbosity;
static int rank, nprocs;       // MPI values
static MPI_Comm comm;


//sets up parallel parameters above
void initParParams(int M_, int N_, int P_, int Q_, int verb) {
  M = M_, N = N_; P = P_, Q = Q_;
  verbosity = verb;
  comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &nprocs);
// 2D process id
  Q0 = rank / P; 
  P0 = rank % P;
  N0 = (N / Q) * Q0;
  M0 = (M / P) * P0;
  N_loc = (Q0 < Q-1)? (N / Q): (N - N0); 
  M_loc = (P0 < P-1)? (M / P): (M - M0); 
} 
//*********************************************************************
// 2D Blocking implementation
//*********************************************************************
// static void updateBoundary(int w, double *u, int ldu) {
//   int i, j, k;
//   MPI_Datatype Slice;
//   MPI_Datatype MSlice;
//   if (Q == 1) { 
//     for (i = 1; i < M_loc+1; i++) {
//       V(u, i, 0) = V(u, i, N_loc);
//       V(u, i, N_loc+1) = V(u, i, 1);
//     }
//   } else {
//         int topProc = (Q0-1+Q)%Q * P +P0;// get rank of top process
//         int downProc = (Q0+1)%Q * P +P0; // ger rank of down process
//         MPI_Type_vector ( w, M_loc , M_loc+2*w , MPI_DOUBLE , & MSlice );
//         MPI_Type_commit ( & MSlice );
//     	if (Q0 == 0){    	  
//           MPI_Send(&V(u, w, N_loc), 1, MSlice, downProc, HALO_TAG, comm);
//           MPI_Recv(&V(u, w, 0), 1, MSlice, topProc, HALO_TAG, comm, 
// 	           MPI_STATUS_IGNORE);
//           MPI_Send(&V(u, w, w), 1, MSlice, topProc, HALO_TAG, comm);
//           MPI_Recv(&V(u, w, N_loc+w), 1, MSlice, downProc, HALO_TAG, comm, 
//             MPI_STATUS_IGNORE);
//   	}else{
// 	     MPI_Recv(&V(u, w, 0), 1, MSlice, topProc, HALO_TAG, comm, 
// 	           MPI_STATUS_IGNORE);
// 	     MPI_Send(&V(u, w, N_loc), 1,MSlice, downProc, HALO_TAG, comm);
//        MPI_Recv(&V(u, w, N_loc+w), 1, MSlice, downProc, HALO_TAG, 
// 	           comm, MPI_STATUS_IGNORE);
//        MPI_Send(&V(u, w, w), 1, MSlice, topProc, HALO_TAG, comm);
//         }
//   }

//   //top and bottom; we can get the corner elements from the top & bottom
//   if (P == 1) {
//     for (j = 0; j < N_loc+2; j++) {
//       V(u, 0, j) = V(u, M_loc, j);
//       V(u, M_loc+1, j) = V(u, 1, j);      
//     }
//   } else {
//      int leftProc = Q0 * P + (P0-1+P) % P; //get the rank of left process
//      int rightProc = Q0 * P + (P0+1) % P;// get the rank of right process
//      MPI_Type_vector ( N_loc + 2*w, w, M_loc+2*w , MPI_DOUBLE , & Slice );
//      MPI_Type_commit ( & Slice );
//      if (P0 == 0){
//       MPI_Send(&V(u, w, 0), 1, Slice, leftProc, HALO_TAG, comm);
//       MPI_Recv(&V(u, M_loc + w, 0), 1, Slice, rightProc, HALO_TAG, comm, 
//              MPI_STATUS_IGNORE);
//       MPI_Send(&V(u, M_loc, 0), 1, Slice, rightProc, HALO_TAG, comm);
//       MPI_Recv(&V(u, 0, 0), 1, Slice, leftProc, HALO_TAG, comm, 
//              MPI_STATUS_IGNORE);

//      }else{
//       MPI_Recv(&V(u, M_loc + w, 0), 1, Slice, rightProc, HALO_TAG, comm, 
//              MPI_STATUS_IGNORE);
//       MPI_Send(&V(u, w, 0), 1, Slice, leftProc, HALO_TAG, comm);
//       MPI_Recv(&V(u, 0, 0), 1, Slice, leftProc, HALO_TAG, comm, 
//              MPI_STATUS_IGNORE);
//       MPI_Send(&V(u, M_loc, 0), 1, Slice, rightProc, HALO_TAG, comm);
      
//       }
//   }

// } 

// 2D Non-Blocking implementation
static void updateBoundary(int w, double *u, int ldu) {
  int i, j, k, q;
  MPI_Datatype Slice;
  MPI_Datatype MSlice;

  if (Q == 1) { 
    for(q = 0; q < w; q++){
      for (i = w; i < M_loc+w; i++) {
        V(u, i, 0+q) = V(u, i, N_loc+q);
        V(u, i, N_loc+q+w) = V(u, i, w+q);
       }
    }
  } else {
        int topProc = (Q0-1+Q)%Q * P + P0;// get rank of top process
        int downProc = (Q0+1)%Q * P + P0; // ger rank of down process
        MPI_Type_vector ( w, M_loc, M_loc+2*w , MPI_DOUBLE , & MSlice );//define MPI vector datatype
        MPI_Type_commit ( & MSlice );
        MPI_Request req_td[4];

        MPI_Isend(&V(u, w, N_loc), 1, MSlice, downProc, HALO_TAG, comm,req_td);
        MPI_Irecv(&V(u, w, 0), 1, MSlice, topProc, HALO_TAG, comm, 
             req_td +1);
        MPI_Isend(&V(u, w, w), 1, MSlice, topProc, HALO_TAG, comm,req_td+2);
        MPI_Irecv(&V(u, w, N_loc+w), 1, MSlice, downProc, HALO_TAG, comm, 
           req_td+3);
        for( k = 0; k < 4; k++){
          MPI_Wait(req_td + k, MPI_STATUS_IGNORE);//wait all send and rcev finish
        }
    }

  if (P == 1) {
    for(q = 0; q < w; q++){
      for (j = 0; j < N_loc+2*w; j++) {
        V(u, 0+q, j) = V(u, M_loc+q, j);
        V(u, M_loc+w+q, j) = V(u, w+q, j);      
      }
    }
  } else {
     int leftProc = Q0 * P + (P0-1+P) % P; //get the rank of left process
     int rightProc = Q0 * P + (P0+1) % P;// get the rank of right process
     MPI_Type_vector ( N_loc + 2*w, w, M_loc+2*w, MPI_DOUBLE , & Slice );//define MPI vector datatype
     MPI_Type_commit ( &Slice );
     MPI_Request req_lr[4];
     MPI_Isend(&V(u, w, 0), 1, Slice, leftProc, HALO_TAG, comm, req_lr);
     MPI_Irecv(&V(u, M_loc + w, 0), 1, Slice, rightProc, HALO_TAG, comm, 
             req_lr+1);
     MPI_Isend(&V(u, M_loc, 0), 1, Slice, rightProc, HALO_TAG, comm, req_lr+2);
     MPI_Irecv(&V(u, 0, 0), 1, Slice, leftProc, HALO_TAG, comm, 
             req_lr+3);
     for( k = 0; k < 4; k++){
          MPI_Wait(req_lr + k, MPI_STATUS_IGNORE);//wait all send and rcev finish
        }  
  }

} //updateBoundary()


// evolve advection over r timesteps, with (u,ldu) containing the local field
void parAdvect(int reps, double *u, int ldu) {
  int r, w = 1;
  for (r = 0; r < reps; r++) {
    updateBoundary(1, u, ldu);
    updateAdvectField(M_loc, N_loc, &V(u,w,w), ldu);

    if (verbosity > 2) {
      char s[64]; sprintf(s, "%d reps: u", r+1);
      printAdvectField(rank, s, M_loc+2, N_loc+2, u, ldu);
    }
  }
}

// overlap communication variant
// overlap only works on 1D gird which P = 1
void parAdvectOverlap(int reps, double *u, int ldu) { 
  int r,j;
  MPI_Datatype MSlice;
  MPI_Request a[4];
  
  double *oldt = malloc(M_loc * sizeof(double));
  double *oldd = malloc(M_loc * sizeof(double));
  double *newt = malloc(M_loc * sizeof(double));
  double *newd = malloc(M_loc * sizeof(double));

  updateBoundary(1, u, ldu); // initially update boundary

  for ( r = 0; r < reps; r++){
    //save old values of top and down cells before update
    memcpy(oldt, &V(u, 1, 1), M_loc* sizeof(double)); 
    memcpy(oldd, &V(u, 1, N_loc), M_loc* sizeof(double));
    // update the top and down cells
    updateAdvectField(M_loc, 1, &V(u, 1, 1), ldu);
    updateAdvectField(M_loc, 1, &V(u, 1, N_loc), ldu);
    // save new values of top and down cells 
    memcpy(newt, &V(u, 1, 1), M_loc* sizeof(double));
    memcpy(newd, &V(u, 1, N_loc), M_loc* sizeof(double));
    // send/ recv the top and down cells to/from neiborhood by non-blocking send/ recv
    int topProc = (Q0-1+Q)%Q * P +P0;
    int downProc = (Q0+1)%Q * P +P0; 
    MPI_Type_vector ( 1, M_loc , M_loc+2, MPI_DOUBLE , & MSlice );
    MPI_Type_commit ( & MSlice );
    MPI_Isend(newd, 1, MSlice, downProc, HALO_TAG, comm,a);
    MPI_Irecv(&V(u, 1, 0), 1, MSlice, topProc, HALO_TAG, comm, 
             a+1);
    MPI_Isend(newt, 1, MSlice, topProc, HALO_TAG, comm,a+2);
    MPI_Irecv(&V(u, 1, N_loc+1), 1, MSlice, downProc, HALO_TAG, comm, 
            a+3);
    // recover old value of top and down cell for rest part updating
    memcpy(&V(u, 1, 1), oldt, M_loc* sizeof(double));
    memcpy(&V(u, 1, N_loc),oldd, M_loc* sizeof(double));
    // update rest part
    updateAdvectField(M_loc, N_loc - 2, &V(u, 1, 2), ldu);
    // recover new value of top and down for next time step
    memcpy(&V(u, 1, 1), newt, M_loc * sizeof(double));
    memcpy(&V(u, 1, N_loc),newd, M_loc * sizeof(double));

    MPI_Wait(a, MPI_STATUS_IGNORE);
    MPI_Wait(a+1, MPI_STATUS_IGNORE);
    MPI_Wait(a+2, MPI_STATUS_IGNORE);
    MPI_Wait(a+3, MPI_STATUS_IGNORE);
    //update for right and left halo
    for (j = 0; j < N_loc+2; j++) {
      V(u, 0, j) = V(u, M_loc, j);
      V(u, M_loc+1, j) = V(u, 1, j);      
    }
  }
}

// wide halo variant
void parAdvectWide(int reps, int w, double *u, int ldu) {
  int r,echo;
  for (r = 0; r < reps; r++){
    if( r % w == 0){
      // communicate every r/w times
       updateBoundary(w, u, ldu);
      for(echo = 1; echo < w+1; echo ++){
        //updatafield w times
        updateAdvectField(M_loc+2*w-2*echo, N_loc+2*w-2*echo, &V(u,echo,echo), ldu);
      }
    }
  }

    if (verbosity > 2) {
      char s[64]; sprintf(s, "%d reps: u", r+1);
      printAdvectField(rank, s, M_loc+2, N_loc+2, u, ldu);
    }
}

// extra optimization variant
void parAdvectExtra(int r, double *u, int ldu) {
}
