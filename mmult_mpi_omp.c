#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/times.h>
#define min(x, y) ((x)<(y)?(x):(y))

double* gen_matrix(int n, int m);
int mmult_omp(double *c, double *a, int aRows, int aCols, double *b, int bRows, int bCols);
int mmult(double *c, double *a, int aRows, int aCols, double *b, int bRows, int bCols);
void compare_matrix(double *a, double *b, int nRows, int nCols);

/** 
    Program to multiply a matrix times a matrix using both
    mpi to distribute the computation among nodes and omp
    to distribute the computation among threads.
*/

int main(int argc, char* argv[])
{
  int nrows, ncols;
  double *aa;	/* the A matrix */
  double *bb;	/* the B matrix */
  double *cc1;	/* A x B computed using the omp-mpi code you write */
  double *cc2;	/* A x B computed using the conventional algorithm */
  int myid, numprocs;
  double starttime, endtime;
  MPI_Status status;
  /* insert other global variables here */
  double *strp_buffer, strp_ret;
  int strp_rows = nrows / numprocs;
  int strp_number;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  if (argc > 1) {
    nrows = atoi(argv[1]);
    ncols = nrows;
    if (myid == 0) {
      // Master Code goes here
      aa = gen_matrix(nrows, ncols);
      bb = gen_matrix(ncols, nrows);
      cc1 = malloc(sizeof(double) * nrows * nrows);
      starttime = MPI_Wtime();
      /* Insert your master code here to store the product into cc1 */

      // Stripe the A matrix and distribute stripes.
      strp_buffer = malloc(sizeof(double) * strp_rows * ncols);
      int strp_sent = 0;

      MPI_Bcast(bb, nrows * ncols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      // Send the evenly divisible stripes; i controls which stripe, j controls position within the stripe
      int i, j, k;
      for(i = 0; i < min(ncols, numprocs); i++) {
         for(j = 0; j < strp_rows * numcols; j++) {
             strp_buffer[j] = aa[(i * strp_rows * ncols) + j];
         }
         MPI_Send(strp_buffer, strp_rows * ncols, MPI_DOUBLE, i+1, i+1, MPI_COMM_WORLD);
         strp_sent++;
      }

      //Handle any remaining rows as its own smaller stripe
      if((strp_sent * strp_rows) < nrows) {
        //Clear the stripe buffer, important for when we receive the smaller stripe once more
        for(j = 0; j < strp_rows * ncols; j++) {
           strp_buffer[j] = 0;
        }

         for(j = 0, k = (i * strp_rows * ncols); k < (nrows * ncols); j++, k++) {
             strp_buffer[j] = aa[k];
         }

         MPI_Send(strp_buffer, j, MPI_DOUBLE, i+1, i+1, MPI_COMM_WORLD);
         strp_sent++;
      }
      // Receive Stripes
      for(i = 0; i < strp_sent; i++) {
         MPI_Recv(&cc1, strp_rows * ncols, MPI_DOUBLE, MPI_ANY_SOURCE,
                  MPI_COMM_WORLD, &status);

      }

      // Assemble final matrix

      endtime = MPI_Wtime();
      printf("%f\n",(endtime - starttime));
      cc2  = malloc(sizeof(double) * nrows * nrows);
      mmult(cc2, aa, nrows, ncols, bb, ncols, nrows);
      compare_matrices(cc2, cc1, nrows, nrows);
    } else {
      // Slave Code goes here
      MPI_Bcast(bb, nrows * ncols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      // Receive my A Stripe
      MPI_Recv(strp_buffer, strp_rows, MPI_DOUBLE, 0, MPI_ANY_TAG,
               MPI_COMM_WORLD, &status);

      strp_number = status.MPI_TAG;
      strp_ret = malloc(sizeof(double) * strp_rows * ncols);
      // Complete personal As x B calculation
#pragma omp parallel
#pragma omp shared(strp_ret) for reduction(+:strp_ret)
      mmult(strp_ret, strp_buffer, strp_rows, ncols, bb, ncols, nrows); 
      // Send calculation back to Master
      MPI_Send(&strp_ret, strp_rows * ncols, MPI_DOUBLE, 0,
               strp_number, MPI_COMM_WORLD);
    }
  } else {
    fprintf(stderr, "Usage matrix_times_vector <size>\n");
  }
  MPI_Finalize();
  return 0;
}
