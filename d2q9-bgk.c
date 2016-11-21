/*
 ** Code to implement a d2q9-bgk lattice boltzmann scheme.
 ** 'd2' inidates a 2-dimensional grid, and
 ** 'q9' indicates 9 velocities per grid cell.
 ** 'bgk' refers to the Bhatnagar-Gross-Krook collision step.
 **
 ** The 'speeds' in each cell are numbered as follows:
 **
 ** 6 2 5
 **  \|/
 ** 3-0-1
 **  /|\
 ** 7 4 8
 **
 ** A 2D grid:
 **
 **           cols
 **       --- --- ---
 **      | D | E | F |
 ** rows  --- --- ---
 **      | A | B | C |
 **       --- --- ---
 **
 ** 'unwrapped' in row major order to give a 1D array:
 **
 **  --- --- --- --- --- ---
 ** | A | B | C | D | E | F |
 **  --- --- --- --- --- ---
 **
 ** Grid indicies are:
 **
 **          ny
 **          ^       cols(jj)
 **          |  ----- ----- -----
 **          | | ... | ... | etc |
 **          |  ----- ----- -----
 ** rows(ii) | | 1,0 | 1,1 | 1,2 |
 **          |  ----- ----- -----
 **          | | 0,0 | 0,1 | 0,2 |
 **          |  ----- ----- -----
 **          ----------------------> nx
 **
 ** Note the names of the input parameter and obstacle files
 ** are passed on the command line, e.g.:
 **
 **   d2q9-bgk.exe input.params obstacles.dat
 **
 ** Be sure to adjust the grid dimensions in the parameter file
 ** if you choose a different obstacle file.
 */

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<sys/time.h>
#include<sys/resource.h>
#include<mpi.h>
#include<string.h>
#define NSPEEDS         9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"

/* struct to hold the parameter values */
typedef struct
{
    int    nx;            /* no. of cells in x-direction */
    int    ny;            /* no. of cells in y-direction */
    int    maxIters;      /* no. of iterations */
    int    reynolds_dim;  /* dimension for Reynolds number */
    double density;       /* density per link */
    double accel;         /* density redistribution */
    double omega;         /* relaxation parameter */
} t_param;

/* struct to hold the 'speed' values */
typedef struct
{
    float speeds[NSPEEDS];
} t_speed;

/*
 ** function prototypes
 */

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed** cells_ptr,
               int** obstacles_ptr, double** av_vels_ptr, int rank, int ncols, int nrows);

/*
 ** The main calculation methods.
 ** timestep calls, in order, the functions:
 ** accelerate_flow(), propagate(), rebound() & collision()
 */
int timestep(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles);
int accelerate_flow(const t_param params, t_speed* cells, int* obstacles);
int propagate(const t_param params, t_speed* cells, t_speed* tmp_cells);
int rebound(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles);
double collision(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles);
int write_values(const t_param params, t_speed* cells, int* obstacles, double* av_vels);
int calc_nrows_from_rank(int rank, int size, int rows);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, t_speed** cells_ptr,
             int** obstacles_ptr, double** av_vels_ptr);

/* Sum all the densities in the grid.
 ** The total should remain constant from one timestep to the next. */
double total_density(const t_param params, t_speed* cells);

/* compute average velocity */
double av_velocity(const t_param params, t_speed* cells, int* obstacles);

/* calculate Reynolds number */
double calc_reynolds(const t_param params, t_speed* cells, int* obstacles);

/* utility functions */
void die(const char* message, const int line, const char* file);
void usage(const char* exe);

/*
 ** main program:
 ** initialise, timestep loop, finalise
 */
int main(int argc, char* argv[])
{
    char*    paramfile = NULL;    /* name of the input parameter file */
    char*    obstaclefile = NULL; /* name of a the input obstacle file */
    t_param  params;              /* struct to hold parameter values */
    t_speed* loc_cells = NULL;
    t_speed* loc_tmp_cells = NULL;
    int*     loc_obstacles = NULL;    /* grid indicating which cells are blocked */
    int*     total_obstacles_grid = NULL;
    double* av_vels   = NULL;     /* a record of the av. velocity computed for each timestep */
    struct timeval timstr;        /* structure to hold elapsed time */
    struct rusage ru;             /* structure to hold CPU time--system and user */
    double tic, toc;              /* doubleing point numbers to calculate elapsed wallclock time */
    double usrtim;                /* doubleing point number to record elapsed user CPU time */
    double systim;                /* doubleing point number to record elapsed system CPU time */
    int tot_cells;
    double tot_u;
    int loc_cells_count;
    double loc_u;
    int kk;
    int ii,jj;             /* row and column indices for the grid */
    //int kk;                /* index for looping over ranks */
    //int start_col,end_col; /* rank dependent looping indices */
    //int iter;              /* index for timestep iterations */
    int rank;              /* the rank of this process */
    int left;              /* the rank of the process to the left */
    int right;             /* the rank of the process to the right */
    int size;              /* number of processes in the communicator */
    int tag = 0;           /* scope for adding extra information to a message */
    MPI_Status status;     /* struct used by MPI_Recv */
    int local_nrows;       /* number of rows apportioned to this rank */
    int local_ncols;       /* number of columns apportioned to this rank */
    float *sendbuf;       /* buffer to hold values to send */
    float *recvbuf;       /* buffer to hold received values */
    
    //int not_perf = 0;
    int* sendbuf_obs;       /* buffer to hold values to send */
    //int* recvbuf_obs;       /* buffer to hold received values */
    
    int difference = 0;
    //double **tmp_u;            /* local temperature grid at time t - 1 */
    //t_speed **tmp_w;            /* local temperature grid at time t     */
    //double *printbuf;      /* buffer to hold values for printing */
    
    //    /* MPI_Init returns once it has started up processes */
    MPI_Init( &argc, &argv );
    //
    //    /* size and rank will become ubiquitous */
    MPI_Comm_size( MPI_COMM_WORLD, &size );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    
    //    if(rank == 0 && params.nx % size != 0) {
    //        if(params.nx % size == 0)
    //            not_perf = 1;
    //    }
    
    /* parse the command line */
    if (argc != 3)
    {
        usage(argv[0]);
    }
    else
    {
        paramfile = argv[1];
        obstaclefile = argv[2];
    }
    ////////////////////////////////////////////////////////////////////////////////////////////
    char   message[1024];  /* message buffer */
    FILE*   fp;            /* file pointer */
    int    xx, yy;         /* generic array indices */
    int    blocked;        /* indicates whether a cell is blocked by an obstacle */
    int    retval;         /* to hold return value for checking */
    
    /* open the parameter file */
    fp = fopen(paramfile, "r");
    
    if (fp == NULL)
    {
        sprintf(message, "could not open input parameter file: %s", paramfile);
        die(message, __LINE__, __FILE__);
    }
    
    /* read in the parameter values */
    retval = fscanf(fp, "%d\n", &(params.nx));
    
    if (retval != 1) die("could not read param file: nx", __LINE__, __FILE__);
    
    retval = fscanf(fp, "%d\n", &(params.ny));
    
    if (retval != 1) die("could not read param file: ny", __LINE__, __FILE__);
    
    retval = fscanf(fp, "%d\n", &(params.maxIters));
    
    if (retval != 1) die("could not read param file: maxIters", __LINE__, __FILE__);
    
    retval = fscanf(fp, "%d\n", &(params.reynolds_dim));
    
    if (retval != 1) die("could not read param file: reynolds_dim", __LINE__, __FILE__);
    
    retval = fscanf(fp, "%lf\n", &(params.density));
    
    if (retval != 1) die("could not read param file: density", __LINE__, __FILE__);
    
    retval = fscanf(fp, "%lf\n", &(params.accel));
    
    if (retval != 1) die("could not read param file: accel", __LINE__, __FILE__);
    
    retval = fscanf(fp, "%lf\n", &(params.omega));
    
    if (retval != 1) die("could not read param file: omega", __LINE__, __FILE__);
    
    /* and close up the file */
    fclose(fp);
    
    local_nrows = calc_nrows_from_rank(rank, size, params.ny);
    local_ncols = params.nx;
    
    if(rank == 0) {
        difference = params.ny - local_nrows*size;
        //printf("%d", difference);
    }
    
    if(rank == size-1) {
        //printf("row %d col %d", local_nrows, local_ncols);
    }
    //int local_nrows * local_ncols = local_ncols * local_nrows;
    int speed_ncols = NSPEEDS * local_ncols;
    //printf("rows: %d, cols: %d", local_nrows, local_ncols);
    //    loc_cells_1D = (t_speed*)malloc(sizeof(t_speed) * (local_nrows * local_ncols));
    //    if (loc_cells_1D == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);
    
    //    if(rank == 0) {
    //        malloc(sizeof(int) * (params.ny * params.nx));
    //    }
    
    loc_cells = (t_speed*)malloc(sizeof(t_speed) * (local_nrows+2) * local_ncols);
    //    for(ii=0;ii<local_nrows+2;ii++) {
    //        loc_cells[ii] = (t_speed*)malloc(sizeof(t_speed) * local_ncols);
    //    }
    if (loc_cells == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);
    loc_tmp_cells = (t_speed*)malloc(sizeof(t_speed) * (local_nrows+2) * local_ncols);
    //    for(ii=0;ii<local_nrows+2;ii++) {
    //        loc_tmp_cells[ii] = (t_speed*)malloc(sizeof(t_speed) * (local_ncols));
    //    }
    
    /* the map of obstacles */
    loc_obstacles = malloc(sizeof(int) * (local_nrows * local_ncols));
    
    if(rank ==0){
        total_obstacles_grid = malloc(sizeof(int) * (params.ny * params.nx));
    }
    
    if (loc_obstacles == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);
    
    /* initialise densities */
    float w5 = params.density * 4.0 / 9.0;
    float w6 = params.density      / 9.0;
    float w7 = params.density      / 36.0;
    
    for (ii = 1; ii < local_nrows + 1; ii++)
    {
        for (jj = 0; jj < local_ncols; jj++)
        {
            /* centre */
            loc_cells[ii*local_ncols + jj].speeds[0] = w5;
            /* axis directions */
            loc_cells[ii*local_ncols + jj].speeds[1] = w6;
            loc_cells[ii*local_ncols + jj].speeds[2] = w6;
            loc_cells[ii*local_ncols + jj].speeds[3] = w6;
            loc_cells[ii*local_ncols + jj].speeds[4] = w6;
            /* diagonals */
            loc_cells[ii*local_ncols + jj].speeds[5] = w7;
            loc_cells[ii*local_ncols + jj].speeds[6] = w7;
            loc_cells[ii*local_ncols + jj].speeds[7] = w7;
            loc_cells[ii*local_ncols + jj].speeds[8] = w7;
            //printf("%d", ii * ncols + jj);
        }
    }
    
    //    for (ii = 0; ii < local_nrows; ii++)
    //    {
    //        for (jj = 0; jj < local_ncols; jj++)
    //        {
    //            loc_obstacles[ii*local_ncols + jj] = 0;
    //        }
    //    }
    
    /* open the obstacle data file */
    fp = fopen(obstaclefile, "r");
    
    if (fp == NULL)
    {
        sprintf(message, "could not open input obstacles file: %s", obstaclefile);
        die(message, __LINE__, __FILE__);
    }
    
    /* read-in the blocked cells list */
    if (rank == 0) {
        while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF)
        {
            /* some checks */
            if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);
            
            if (xx < 0 || xx > params.nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);
            
            if (yy < 0 || yy > params.ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);
            
            if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);
            
            /* assign to array */
            total_obstacles_grid[yy * params.nx + xx] = blocked;
            //printf("%d ", total_obstacles_grid[yy * params.nx + xx]);
        }
    }
    
    if(rank == 0) {
        for(ii = 0; ii<params.ny; ii++) {
            for(jj = 0; jj<params.nx; jj++) {
                if(total_obstacles_grid[ii*params.nx + jj] != 0 && total_obstacles_grid[ii*params.nx + jj] != 1)
                    total_obstacles_grid[ii*params.nx + jj] = 0;
            }
        }
    }
    
    /* and close the file */
    fclose(fp);
    av_vels = (double*)malloc(sizeof(double) * params.maxIters);
    
    /////////////////////////////////////////////////////////////////////////////////////////
    
    right = (rank + 1) % size;
    left = (rank == 0) ? (rank + size - 1) : (rank - 1);
    
    sendbuf = (float*)malloc(sizeof(float) * speed_ncols);
    recvbuf = (float*)malloc(sizeof(float) * speed_ncols);
    
    sendbuf_obs = (int*)malloc(sizeof(int) * local_nrows * local_ncols);
    //printf("%d ", sizeof(int) * local_nrows * local_ncols);
    //printf("total : %d", sizeof(int) * params.nx * params.ny);
    //recvbuf_obs = (int*)malloc(sizeof(int) * local_nrows * local_ncols);
    
    int* spec_sendbuf_obs;
    
    if(rank == 0) {
        spec_sendbuf_obs = (int*)malloc(sizeof(int) * (local_nrows + difference) * local_ncols);
    }
    
    if(rank != size -1) {
        if(rank == 0) {
            for(kk = 0; kk<size - 1; kk++) {
                for(ii = 0; ii<local_nrows; ii++) {
                    for(jj = 0; jj<local_ncols; jj++) {
                        sendbuf_obs[ii*local_ncols + jj] = total_obstacles_grid[ii*params.nx + jj + kk*local_nrows * local_ncols];
                    }
                }
                
                MPI_Send(sendbuf_obs,local_nrows * local_ncols,MPI_INT,kk,tag,MPI_COMM_WORLD);
            }
        }
        MPI_Recv(loc_obstacles,local_nrows * local_ncols,MPI_INT,0,tag,MPI_COMM_WORLD,&status);
        //printf("this is ran");
    }

    if(size != 1) {
        if((rank == 0) || (rank == size - 1)) {
            if(rank == 0) {
                for(ii = 0; ii<local_nrows + difference; ii++) {
                    for(jj = 0; jj<local_ncols; jj++) {
                        spec_sendbuf_obs[ii*local_ncols + jj] = total_obstacles_grid[ii*params.nx + jj + (size - 1)*local_nrows * local_ncols];
                    }
                }
                MPI_Send(spec_sendbuf_obs,(local_nrows + difference) * local_ncols,MPI_INT,size - 1,tag,MPI_COMM_WORLD);
            } else if (rank == size - 1) {
                MPI_Recv(loc_obstacles,local_nrows * local_ncols,MPI_INT,0,tag,MPI_COMM_WORLD,&status);
            }
            //printf("this is ran");
        }
    }
//
    if(size == 1) {
        //for(kk = 0; kk<1; kk++) {
            for(ii = 0; ii<local_nrows; ii++) {
                for(jj = 0; jj<local_ncols; jj++) {
                    sendbuf_obs[ii*local_ncols + jj] = total_obstacles_grid[ii*params.nx + jj];
                }
            }
            
            MPI_Send(sendbuf_obs,local_nrows * local_ncols,MPI_INT,0,tag,MPI_COMM_WORLD);
        //}
        
        MPI_Recv(loc_obstacles,local_nrows * local_ncols,MPI_INT,0,tag,MPI_COMM_WORLD,&status);
        //loc_obstacles = total_obstacles_grid;
    }
////
    const float c_sq = 3.0; /* square of speed of sound */
    const float w0 = 4.0 / 9.0;  /* weighting factor */
    const float w1 = 1.0 / 9.0;  /* weighting factor */
    
    const float two_c_sq_c_sq = 4.5;
    const float two_c_sq = 1.5;
    
    //const int dimension = params.ny*params.nx;
    float w3 = params.density * params.accel / 9.0;
    float w4 = params.density * params.accel / 36.0;
    
    int tt;
    if(rank == 0) {
        gettimeofday(&timstr, NULL);
        tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
    }
    //
    for (tt = 0; tt < params.maxIters; tt++)
    {
        if(rank == size - 1) {
            ii = local_nrows - 1;
            for (jj=0;jj<local_ncols;jj++)
            {
                int index = ii*local_ncols + jj;
                if (!loc_obstacles[(local_nrows-2)*local_ncols + jj]
                    && (loc_cells[index].speeds[3] - w3) > 0.0
                    && (loc_cells[index].speeds[6] - w4) > 0.0
                    && (loc_cells[index].speeds[7] - w4) > 0.0)
                {
                    loc_cells[index].speeds[1] += w3;
                    loc_cells[index].speeds[5] += w4;
                    loc_cells[index].speeds[8] += w4;
                    loc_cells[index].speeds[3] -= w3;
                    loc_cells[index].speeds[6] -= w4;
                    loc_cells[index].speeds[7] -= w4;
                }
            }
        }
        
        //        for(jj=0;jj<local_ncols;jj++) {
        //            *(sendbuf+jj*NSPEEDS) = loc_cells[local_ncols + jj].speeds[0];
        //            *(sendbuf+jj*NSPEEDS + 1) = loc_cells[local_ncols + jj].speeds[1];
        //            *(sendbuf+jj*NSPEEDS + 2) = loc_cells[local_ncols + jj].speeds[2];
        //            *(sendbuf+jj*NSPEEDS + 3) = loc_cells[local_ncols + jj].speeds[3];
        //            *(sendbuf+jj*NSPEEDS + 4) = loc_cells[local_ncols + jj].speeds[4];
        //            *(sendbuf+jj*NSPEEDS + 5) = loc_cells[local_ncols + jj].speeds[5];
        //            *(sendbuf+jj*NSPEEDS + 6) = loc_cells[local_ncols + jj].speeds[6];
        //            *(sendbuf+jj*NSPEEDS + 7) = loc_cells[local_ncols + jj].speeds[7];
        //            *(sendbuf+jj*NSPEEDS + 8) = loc_cells[local_ncols + jj].speeds[8];
        //        }
        memcpy(sendbuf, loc_cells+local_ncols, sizeof(float) * NSPEEDS * local_ncols);
        MPI_Sendrecv(sendbuf, speed_ncols, MPI_FLOAT, left, tag,
                     recvbuf, speed_ncols, MPI_FLOAT, right, tag,
                     MPI_COMM_WORLD, &status);
        //
        //        for(jj=0;jj<local_ncols;jj++) {
        //            loc_cells[(local_nrows+1)*local_ncols + jj].speeds[0] = recvbuf[jj*NSPEEDS];
        //            loc_cells[(local_nrows+1)*local_ncols + jj].speeds[1] = recvbuf[jj*NSPEEDS + 1];
        //            loc_cells[(local_nrows+1)*local_ncols + jj].speeds[2] = recvbuf[jj*NSPEEDS + 2];
        //            loc_cells[(local_nrows+1)*local_ncols + jj].speeds[3] = recvbuf[jj*NSPEEDS + 3];
        //            loc_cells[(local_nrows+1)*local_ncols + jj].speeds[4] = recvbuf[jj*NSPEEDS + 4];
        //            loc_cells[(local_nrows+1)*local_ncols + jj].speeds[5] = recvbuf[jj*NSPEEDS + 5];
        //            loc_cells[(local_nrows+1)*local_ncols + jj].speeds[6] = recvbuf[jj*NSPEEDS + 6];
        //            loc_cells[(local_nrows+1)*local_ncols + jj].speeds[7] = recvbuf[jj*NSPEEDS + 7];
        //            loc_cells[(local_nrows+1)*local_ncols + jj].speeds[8] = recvbuf[jj*NSPEEDS + 8];
        //        }
        memcpy(loc_cells + (local_nrows+1)*local_ncols, recvbuf, sizeof(float) * NSPEEDS * local_ncols);
        
        //        for(jj=0;jj<local_ncols;jj++) {
        //            *(sendbuf+jj*NSPEEDS) = loc_cells[local_nrows*local_ncols + jj].speeds[0];
        //            *(sendbuf+jj*NSPEEDS + 1) = loc_cells[local_nrows*local_ncols + jj].speeds[1];
        //            *(sendbuf+jj*NSPEEDS + 2) = loc_cells[local_nrows*local_ncols + jj].speeds[2];
        //            *(sendbuf+jj*NSPEEDS + 3) = loc_cells[local_nrows*local_ncols + jj].speeds[3];
        //            *(sendbuf+jj*NSPEEDS + 4) = loc_cells[local_nrows*local_ncols + jj].speeds[4];
        //            *(sendbuf+jj*NSPEEDS + 5) = loc_cells[local_nrows*local_ncols + jj].speeds[5];
        //            *(sendbuf+jj*NSPEEDS + 6) = loc_cells[local_nrows*local_ncols + jj].speeds[6];
        //            *(sendbuf+jj*NSPEEDS + 7) = loc_cells[local_nrows*local_ncols + jj].speeds[7];
        //            *(sendbuf+jj*NSPEEDS + 8) = loc_cells[local_nrows*local_ncols + jj].speeds[8];
        //        }
        memcpy(sendbuf, loc_cells+local_nrows*local_ncols, sizeof(float) * NSPEEDS * local_ncols);
        MPI_Sendrecv(sendbuf, speed_ncols, MPI_FLOAT, right, tag,
                     recvbuf, speed_ncols, MPI_FLOAT, left, tag,
                     MPI_COMM_WORLD, &status);
        
        //        for(jj=0;jj<local_ncols;jj++) {
        //            loc_cells[jj].speeds[0] = recvbuf[jj*NSPEEDS];
        //            loc_cells[jj].speeds[1] = recvbuf[jj*NSPEEDS + 1];
        //            loc_cells[jj].speeds[2] = recvbuf[jj*NSPEEDS + 2];
        //            loc_cells[jj].speeds[3] = recvbuf[jj*NSPEEDS + 3];
        //            loc_cells[jj].speeds[4] = recvbuf[jj*NSPEEDS + 4];
        //            loc_cells[jj].speeds[5] = recvbuf[jj*NSPEEDS + 5];
        //            loc_cells[jj].speeds[6] = recvbuf[jj*NSPEEDS + 6];
        //            loc_cells[jj].speeds[7] = recvbuf[jj*NSPEEDS + 7];
        //            loc_cells[jj].speeds[8] = recvbuf[jj*NSPEEDS + 8];
        //        }
        
        memcpy(loc_cells, recvbuf, sizeof(float) * NSPEEDS * local_ncols);
        
        loc_u = 0.0;
        loc_cells_count = 0;
        tot_cells = 0;
        tot_u = 0.0;
        
        for (ii = 1; ii<local_nrows+1; ii++){
            for(jj=0;jj<local_ncols;jj++) {
                float* tmp_speed = loc_tmp_cells[ii*local_ncols + jj].speeds;
                
                int y_n = ii + 1;
                int x_e = (jj == local_ncols -1) ? 0 : (jj + 1);
                int y_s = ii - 1;
                int x_w = (jj == 0) ? (jj + local_ncols - 1) : (jj - 1);;
                
                tmp_speed[0] = loc_cells[ii*local_ncols + jj].speeds[0];
                tmp_speed[1] = loc_cells[ii*local_ncols + x_w].speeds[1];
                tmp_speed[2] = loc_cells[y_s*local_ncols + jj].speeds[2];
                tmp_speed[3] = loc_cells[ii*local_ncols + x_e].speeds[3];
                tmp_speed[4] = loc_cells[y_n*local_ncols + jj].speeds[4];
                tmp_speed[5] = loc_cells[y_s*local_ncols + x_w].speeds[5];
                tmp_speed[6] = loc_cells[y_s*local_ncols + x_e].speeds[6];
                tmp_speed[7] = loc_cells[y_n*local_ncols + x_e].speeds[7];
                tmp_speed[8] = loc_cells[y_n*local_ncols + x_w].speeds[8];
                
            }
        }
        
        
        for (ii = 1; ii<local_nrows + 1; ii++){
            for(jj=0;jj<local_ncols;jj++) {
                
                float* current_speed = loc_cells[ii*local_ncols + jj].speeds;
                float* tmp_speed = loc_tmp_cells[ii*local_ncols + jj].speeds;
                
                if (!loc_obstacles[(ii - 1)*local_ncols + jj])
                {
                    double u_x = (tmp_speed[1]
                                  + tmp_speed[5]
                                  + tmp_speed[8]
                                  - (tmp_speed[3]
                                     + tmp_speed[6]
                                     + tmp_speed[7]))
                    / (tmp_speed[0] + tmp_speed[1] + tmp_speed[2] + tmp_speed[3] + tmp_speed[4] + tmp_speed[5] + tmp_speed[6] + tmp_speed[7] + tmp_speed[8]);
                    
                    double u_y = (tmp_speed[2]
                                  + tmp_speed[5]
                                  + tmp_speed[6]
                                  - (tmp_speed[4]
                                     + tmp_speed[7]
                                     + tmp_speed[8]))
                    / (tmp_speed[0] + tmp_speed[1] + tmp_speed[2] + tmp_speed[3] + tmp_speed[4] + tmp_speed[5] + tmp_speed[6] + tmp_speed[7] + tmp_speed[8]);
                    
                    double u = u_x*u_x + u_y*u_y;
                    loc_u += sqrt(u);
                    double local_density_mult = u * (two_c_sq);
                    double w1_loc = (w1 * (tmp_speed[0] + tmp_speed[1] + tmp_speed[2] + tmp_speed[3] + tmp_speed[4] + tmp_speed[5] + tmp_speed[6] + tmp_speed[7] + tmp_speed[8]));
                    double w2_loc = w1_loc/4.0;
                    
                    
                    *(current_speed) = tmp_speed[0] + params.omega * ((w0* (tmp_speed[0] + tmp_speed[1] + tmp_speed[2] + tmp_speed[3] + tmp_speed[4] + tmp_speed[5] + tmp_speed[6] + tmp_speed[7] + tmp_speed[8])
                                                                       * (1.0 - local_density_mult)) - tmp_speed[0]);
                    *(current_speed+1) = tmp_speed[1] + params.omega * ((w1_loc * (1.0 + u_x * c_sq
                                                                                   + (u_x * u_x) * (two_c_sq_c_sq)
                                                                                   - local_density_mult)) - tmp_speed[1]);
                    *(current_speed+2) = tmp_speed[2] + params.omega * ((w1_loc * (1.0 + u_y * c_sq
                                                                                   + (u_y * u_y) * (two_c_sq_c_sq)
                                                                                   - local_density_mult)) - tmp_speed[2]);
                    *(current_speed+3) = tmp_speed[3] + params.omega * ((w1_loc * (1.0 + (-u_x) * c_sq
                                                                                   + ((-u_x) * (-u_x)) * (two_c_sq_c_sq)
                                                                                   - local_density_mult)) - tmp_speed[3]);
                    *(current_speed+4) = tmp_speed[4] + params.omega * ((w1_loc * (1.0 + (-u_y) * c_sq
                                                                                   + ((-u_y) * (-u_y)) * (two_c_sq_c_sq)
                                                                                   - local_density_mult)) - tmp_speed[4]);
                    *(current_speed+5) = tmp_speed[5] + params.omega * ((w2_loc * (1.0 + (u_x + u_y) * c_sq
                                                                                   + ((u_x + u_y) * (u_x + u_y)) * (two_c_sq_c_sq)
                                                                                   - local_density_mult)) - tmp_speed[5]);
                    *(current_speed+6) = tmp_speed[6] + params.omega * ((w2_loc * (1.0 + (- u_x + u_y) * c_sq
                                                                                   + ((- u_x + u_y) * (- u_x + u_y)) * (two_c_sq_c_sq)
                                                                                   - local_density_mult)) - tmp_speed[6]);
                    *(current_speed+7) = tmp_speed[7] + params.omega * ((w2_loc * (1.0 + (- u_x - u_y) * c_sq
                                                                                   + ((- u_x - u_y) * (- u_x - u_y)) * (two_c_sq_c_sq)
                                                                                   - local_density_mult)) - tmp_speed[7]);
                    *(current_speed+8) = tmp_speed[8] + params.omega * ((w2_loc * (1.0 + ( u_x - u_y) * c_sq
                                                                                   + (( u_x - u_y) * ( u_x - u_y)) * (two_c_sq_c_sq)
                                                                                   - local_density_mult)) - tmp_speed[8]);
                    
                    //printf(" %d index, %d rank, %d row, %d col : %f\n", (ii - 1)*params.nx + jj +rank*local_ncols*(local_nrows -2), rank, ii - 1, jj, sqrt(u));
                    // printf(" %d index, %d rank, %d row, %d col : %f\n", (ii - 1)*params.nx + jj +rank*local_ncols*(local_nrows),rank, ii - 1, jj);
                }else {
                    
                    *(current_speed+1) = tmp_speed[3];
                    *(current_speed+2) = tmp_speed[4];
                    *(current_speed+3) = tmp_speed[1];
                    *(current_speed+4) = tmp_speed[2];
                    *(current_speed+5) = tmp_speed[7];
                    *(current_speed+6) = tmp_speed[8];
                    *(current_speed+7) = tmp_speed[5];
                    *(current_speed+8) = tmp_speed[6];
                    loc_cells_count++;
                }
            }
        }
        
        MPI_Reduce(&loc_cells_count, &tot_cells, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&loc_u, &tot_u, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        
        if (rank == 0)
            av_vels[tt] = tot_u / (double)(params.ny*params.nx-tot_cells);
    }
    //
    if(rank ==0) {
        gettimeofday(&timstr, NULL);
        toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
        getrusage(RUSAGE_SELF, &ru);
        timstr = ru.ru_utime;
        usrtim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
        timstr = ru.ru_stime;
        systim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
    }
    t_speed* total_cells_grid = NULL;
    
    if(rank == 0) {
        total_cells_grid = (t_speed*)malloc(((sizeof(t_speed)) * params.nx * params.ny));
    }
    //    //combine all the grids into total_cells_grid
    
    int mult;
    float* tot_speed;
    double total;
    if (rank != size - 1) {
        for(ii = 1; ii<local_nrows+1; ii++) {
            for(jj=0;jj<local_ncols;jj++) {
                mult = jj*NSPEEDS;
                float* loc_speed = loc_cells[ii*local_ncols + jj].speeds;
                *(sendbuf+mult) = loc_speed[0];
                *(sendbuf+mult + 1) = loc_speed[1];
                *(sendbuf+mult + 2) = loc_speed[2];
                *(sendbuf+mult + 3) = loc_speed[3];
                *(sendbuf+mult + 4) = loc_speed[4];
                *(sendbuf+mult + 5) = loc_speed[5];
                *(sendbuf+mult + 6) = loc_speed[6];
                *(sendbuf+mult + 7) = loc_speed[7];
                *(sendbuf+mult + 8) = loc_speed[8];
                //                printf(" %d index, %d rank, %d row, %d col : %f\n", (ii - 1)*params.nx + jj +rank*local_ncols*(local_nrows), rank, ii - 1, jj, loc_speed[0] + loc_speed[1] + loc_speed[2] + loc_speed[3] + loc_speed[4] + loc_speed[5] + loc_speed[6] + loc_speed[7] + loc_speed[8]);
                //                total += loc_speed[0] + loc_speed[1] + loc_speed[2] + loc_speed[3] + loc_speed[4] + loc_speed[5] + loc_speed[6] + loc_speed[7] + loc_speed[8];
            }
            MPI_Ssend(sendbuf,speed_ncols,MPI_FLOAT,0,tag,MPI_COMM_WORLD);
            
            if (rank == 0) {
                for(kk = 0; kk<size-1; kk++) {
                    MPI_Recv(recvbuf,speed_ncols,MPI_FLOAT,kk,tag,MPI_COMM_WORLD,&status);
                    for(jj=0;jj<local_ncols;jj++) {
                        tot_speed = total_cells_grid[(ii-1)*params.nx + jj +kk * local_nrows * local_ncols].speeds;
                        mult = jj*NSPEEDS;
                        tot_speed[0] = recvbuf[mult];
                        tot_speed[1] = recvbuf[mult + 1];
                        tot_speed[2] = recvbuf[mult + 2];
                        tot_speed[3] = recvbuf[mult + 3];
                        tot_speed[4] = recvbuf[mult + 4];
                        tot_speed[5] = recvbuf[mult + 5];
                        tot_speed[6] = recvbuf[mult + 6];
                        tot_speed[7] = recvbuf[mult + 7];
                        tot_speed[8] = recvbuf[mult + 8];
                        total += tot_speed[0] + tot_speed[1] + tot_speed[2] + tot_speed[3] + tot_speed[4] + tot_speed[5] + tot_speed[6] + tot_speed[7] + tot_speed[8];
                        //printf("%d\n", (ii-1)*params.nx + jj + kk * local_nrows * local_ncols);
                    }
                }
            }
            
        }
    }
    if (size != 1) {
        if ((rank == size - 1) || (rank == 0)) {
            if (rank == size - 1) {
                for(ii = 1; ii<local_nrows+1; ii++) {
                    for(jj=0;jj<local_ncols;jj++) {
                        mult = jj*NSPEEDS;
                        float* loc_speed = loc_cells[ii*local_ncols + jj].speeds;
                        *(sendbuf+mult) = loc_speed[0];
                        *(sendbuf+mult + 1) = loc_speed[1];
                        *(sendbuf+mult + 2) = loc_speed[2];
                        *(sendbuf+mult + 3) = loc_speed[3];
                        *(sendbuf+mult + 4) = loc_speed[4];
                        *(sendbuf+mult + 5) = loc_speed[5];
                        *(sendbuf+mult + 6) = loc_speed[6];
                        *(sendbuf+mult + 7) = loc_speed[7];
                        *(sendbuf+mult + 8) = loc_speed[8];
                        //printf(" %d index, %d rank, %d row, %d col : %f\n", (ii - 1)*params.nx + jj +rank*local_ncols*(local_nrows-difference), rank, ii - 1, jj, loc_speed[0] + loc_speed[1] + loc_speed[2] + loc_speed[3] + loc_speed[4] + loc_speed[5] + loc_speed[6] + loc_speed[7] + loc_speed[8]);
                        //                        total += loc_speed[0] + loc_speed[1] + loc_speed[2] + loc_speed[3] + loc_speed[4] + loc_speed[5] + loc_speed[6] + loc_speed[7] + loc_speed[8];
                    }
                    MPI_Ssend(sendbuf,speed_ncols,MPI_FLOAT,0,tag,MPI_COMM_WORLD);
                }
            } else {
                for(ii = 1; ii<local_nrows + 1 + difference; ii++) {
                    MPI_Recv(recvbuf,speed_ncols,MPI_FLOAT, size - 1,tag,MPI_COMM_WORLD,&status);
                    for(jj=0;jj<local_ncols;jj++) {
                        tot_speed = total_cells_grid[(ii-1)*params.nx + jj + (size - 1) * local_nrows * local_ncols].speeds;
                        mult = jj*NSPEEDS;
                        tot_speed[0] = recvbuf[mult];
                        tot_speed[1] = recvbuf[mult + 1];
                        tot_speed[2] = recvbuf[mult + 2];
                        tot_speed[3] = recvbuf[mult + 3];
                        tot_speed[4] = recvbuf[mult + 4];
                        tot_speed[5] = recvbuf[mult + 5];
                        tot_speed[6] = recvbuf[mult + 6];
                        tot_speed[7] = recvbuf[mult + 7];
                        tot_speed[8] = recvbuf[mult + 8];
                        total += tot_speed[0] + tot_speed[1] + tot_speed[2] + tot_speed[3] + tot_speed[4] + tot_speed[5] + tot_speed[6] + tot_speed[7] + tot_speed[8];
                        // printf("%d\n", (ii-1)*params.nx + jj + (size - 1) * (local_nrows) * local_ncols);
                    }
                    
                }
            }
        }
    }
    else {
        int mult;
        for(ii = 1; ii<local_nrows+1; ii++) {
            for(jj=0;jj<local_ncols;jj++) {
                mult = jj*NSPEEDS;
                float* loc_speed = loc_cells[ii*local_ncols + jj].speeds;
                *(sendbuf+mult) = loc_speed[0];
                *(sendbuf+mult + 1) = loc_speed[1];
                *(sendbuf+mult + 2) = loc_speed[2];
                *(sendbuf+mult + 3) = loc_speed[3];
                *(sendbuf+mult + 4) = loc_speed[4];
                *(sendbuf+mult + 5) = loc_speed[5];
                *(sendbuf+mult + 6) = loc_speed[6];
                *(sendbuf+mult + 7) = loc_speed[7];
                *(sendbuf+mult + 8) = loc_speed[8];
            }
            MPI_Ssend(sendbuf,speed_ncols,MPI_FLOAT,0,tag,MPI_COMM_WORLD);
            
            if (rank == 0) {
                for(kk = 0; kk<size; kk++) {
                    MPI_Recv(recvbuf,speed_ncols,MPI_FLOAT,kk,tag,MPI_COMM_WORLD,&status);
                    for(jj=0;jj<local_ncols;jj++) {
                        //index = (ii-1)*params.nx + jj +kk*local_ncols*local_nrows;
                        float* tot_speed = total_cells_grid[(ii-1)*params.nx + jj +kk*local_nrows * local_ncols].speeds;
                        mult = jj*NSPEEDS;
                        tot_speed[0] = recvbuf[mult];
                        tot_speed[1] = recvbuf[mult + 1];
                        tot_speed[2] = recvbuf[mult + 2];
                        tot_speed[3] = recvbuf[mult + 3];
                        tot_speed[4] = recvbuf[mult + 4];
                        tot_speed[5] = recvbuf[mult + 5];
                        tot_speed[6] = recvbuf[mult + 6];
                        tot_speed[7] = recvbuf[mult + 7];
                        tot_speed[8] = recvbuf[mult + 8];
                    }
                }
            }
            
        }
        
    }
    //printf("%f ", total);
    
    //
    if (rank == 0) {
        //
        //        for(tt=0; tt<params.maxIters; tt++){
        //#ifdef DEBUG
        //            printf("==timestep: %d==\n", tt);
        //            printf("av velocity: %.12E\n", av_vels[tt]);
        //            printf("tot density: %.12E\n", total_density(params, total_cells_grid));
        //#endif
        //        }
        printf("==done==\n");
        printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, total_cells_grid, total_obstacles_grid));
        printf("Elapsed time:\t\t\t%.6lf (s)\n", toc - tic);
        printf("Elapsed user CPU time:\t\t%.6lf (s)\n", usrtim);
        printf("Elapsed system CPU time:\t%.6lf (s)\n", systim);
        write_values(params, total_cells_grid, total_obstacles_grid, av_vels);
        //finalise(&params, &total_cells_grid, &total_obstacles_grid, &av_vels);
    }
    ////
    MPI_Finalize();
    return EXIT_SUCCESS;
}

int calc_nrows_from_rank(int rank, int size, int rows)
{
    int nrows;
    
    nrows = rows / size;       /* integer division */
    if ((rows % size) != 0) {  /* if there is a remainder */
        if (rank == size - 1)
            nrows += rows % size;  /* add remainder to last rank */
    }
    
    return nrows;
}

double av_velocity(const t_param params, t_speed* cells, int* obstacles)
{
    int    tot_cells = 0;  /* no. of cells used in calculation */
    double tot_u = 0.0;          /* accumulated magnitudes of velocity for each cell */
    
    /* initialise */
    int ii;
    //int jj;
    
    for (ii = 0; ii < params.ny*params.nx; ii++)
    {
        /* ignore occupied cells */
        if (!obstacles[ii])
        {
            /* local density total */
            double u_x = 0.0;
            float* current_speed = cells[ii].speeds;
            u_x = (current_speed[1]
                   + current_speed[5]
                   + current_speed[8]
                   - (current_speed[3]
                      + current_speed[6]
                      + current_speed[7]))
            / (current_speed[0] + current_speed[1] + current_speed[2] + current_speed[3] + current_speed[4] + current_speed[5] + current_speed[6] + current_speed[7] + current_speed[8]);
            
            //            printf("%f ", current_speed[0]);
            /* compute y velocity component */
            double u_y = (current_speed[2]
                          + current_speed[5]
                          + current_speed[6]
                          - (current_speed[4]
                             + current_speed[7]
                             + current_speed[8]))
            / (current_speed[0] + current_speed[1] + current_speed[2] + current_speed[3] + current_speed[4] + current_speed[5] + current_speed[6] + current_speed[7] + current_speed[8]);
            /* accumulate the norm of x- and y- velocity components */
            double u = u_x*u_x + u_y*u_y;
            tot_u += sqrt(u);
            
            //printf("%d : %.10f\n", ii, sqrt(u));
        } else
            ++tot_cells;
        //printf("%d ", tot);
    }
    //printf("%d ", tot_cells);
    
    // printf("%f ", tot_u / (double)(params.ny*params.nx-tot_cells));
    return tot_u / (double)(params.ny*params.nx-tot_cells);
}

int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed** cells_ptr,
               int** obstacles_ptr, double** av_vels_ptr, int rank, int ncols, int nrows)
{
    char   message[1024];  /* message buffer */
    FILE*   fp;            /* file pointer */
    int    xx, yy;         /* generic array indices */
    int    blocked;        /* indicates whether a cell is blocked by an obstacle */
    int    retval;         /* to hold return value for checking */
    
    /* open the parameter file */
    fp = fopen(paramfile, "r");
    
    if (fp == NULL)
    {
        sprintf(message, "could not open input parameter file: %s", paramfile);
        die(message, __LINE__, __FILE__);
    }
    
    /* read in the parameter values */
    retval = fscanf(fp, "%d\n", &(params->nx));
    
    if (retval != 1) die("could not read param file: nx", __LINE__, __FILE__);
    
    retval = fscanf(fp, "%d\n", &(params->ny));
    
    if (retval != 1) die("could not read param file: ny", __LINE__, __FILE__);
    
    retval = fscanf(fp, "%d\n", &(params->maxIters));
    
    if (retval != 1) die("could not read param file: maxIters", __LINE__, __FILE__);
    
    retval = fscanf(fp, "%d\n", &(params->reynolds_dim));
    
    if (retval != 1) die("could not read param file: reynolds_dim", __LINE__, __FILE__);
    
    retval = fscanf(fp, "%lf\n", &(params->density));
    
    if (retval != 1) die("could not read param file: density", __LINE__, __FILE__);
    
    retval = fscanf(fp, "%lf\n", &(params->accel));
    
    if (retval != 1) die("could not read param file: accel", __LINE__, __FILE__);
    
    retval = fscanf(fp, "%lf\n", &(params->omega));
    
    if (retval != 1) die("could not read param file: omega", __LINE__, __FILE__);
    
    /* and close up the file */
    fclose(fp);
    
    *cells_ptr = (t_speed*)malloc(sizeof(t_speed) * (nrows * ncols));
    if (*cells_ptr == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);
    
    /* the map of obstacles */
    *obstacles_ptr = malloc(sizeof(int) * (nrows * ncols));
    
    if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);
    
    /* initialise densities */
    double w0 = params->density * 4.0 / 9.0;
    double w1 = params->density      / 9.0;
    double w2 = params->density      / 36.0;
    
    int ii;
    int jj;
    
    for (ii = 0; ii < nrows; ii++)
    {
        for (jj = 0; jj < ncols; jj++)
        {
            /* centre */
            (*cells_ptr)[ii * ncols + jj].speeds[0] = w0;
            /* axis directions */
            (*cells_ptr)[ii * ncols + jj].speeds[1] = w1;
            (*cells_ptr)[ii * ncols + jj].speeds[2] = w1;
            (*cells_ptr)[ii * ncols + jj].speeds[3] = w1;
            (*cells_ptr)[ii * ncols + jj].speeds[4] = w1;
            /* diagonals */
            (*cells_ptr)[ii * ncols + jj].speeds[5] = w2;
            (*cells_ptr)[ii * ncols + jj].speeds[6] = w2;
            (*cells_ptr)[ii * ncols + jj].speeds[7] = w2;
            (*cells_ptr)[ii * ncols + jj].speeds[8] = w2;
            //printf("%d", ii * ncols + jj);
        }
    }
    //printf("%d", ii * ncols + jj);
    /* first set all cells in obstacle array to zero */
    for (ii = 0; ii < nrows; ii++)
    {
        for (jj = 0; jj < ncols; jj++)
        {
            (*obstacles_ptr)[ii*ncols + jj] = 0;
        }
    }
    
    /* open the obstacle data file */
    fp = fopen(obstaclefile, "r");
    
    if (fp == NULL)
    {
        sprintf(message, "could not open input obstacles file: %s", obstaclefile);
        die(message, __LINE__, __FILE__);
    }
    
    /* read-in the blocked cells list */
    while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF)
    {
        /* some checks */
        if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);
        
        if (xx < 0 || xx > params->nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);
        
        if (yy < 0 || yy > params->ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);
        
        if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);
        
        /* assign to array */
        if(yy > nrows || xx > ncols) {
            break;
        } else {
            (*obstacles_ptr)[yy * ncols + xx] = blocked;
        }
    }
    
    /* and close the file */
    fclose(fp);
    
    /*
     ** Allocate memory.
     **
     ** Remember C is pass-by-value, so we need to
     ** pass pointers into the initialise function.
     **
     ** NB we are allocating a 1D array, so that the
     ** memory will be contiguous.  We still want to
     ** index this memory as if it were a (row major
     ** ordered) 2D array, however.  We will perform
     ** some arithmetic using the row and column
     ** coordinates, inside the square brackets, when
     ** we want to access elements of this array.
     **
     ** Note also that we are using a structure to
     ** hold an array of 'speeds'.  We will allocate
     ** a 1D array of these structs.
     */
    // printf("I got here");
    /* main grid */
    /* open the obstacle data file */
    
    /* and close the file */
    /*
     ** allocate space to hold a record of the avarage velocities computed
     ** at each timestep
     */
    *av_vels_ptr = (double*)malloc(sizeof(double) * params->maxIters);
    
    return EXIT_SUCCESS;
}

int finalise(const t_param* params, t_speed** cells_ptr,
             int** obstacles_ptr, double** av_vels_ptr)
{
    /*
     ** free up allocated memory
     */
    free(*cells_ptr);
    *cells_ptr = NULL;
    
    free(*obstacles_ptr);
    *obstacles_ptr = NULL;
    
    free(*av_vels_ptr);
    *av_vels_ptr = NULL;
    
    return EXIT_SUCCESS;
}


double calc_reynolds(const t_param params, t_speed* cells, int* obstacles)
{
    const double viscosity = 1.0 / 6.0 * (2.0 / params.omega - 1.0);
    //printf("%.10f\n", av_velocity(params, cells, obstacles));
    return av_velocity(params, cells, obstacles) * params.reynolds_dim / viscosity;
}

double total_density(const t_param params, t_speed* cells)
{
    double total = 0.0;  /* accumulator */
    
    int ii, jj, kk;
    //#pragma omp parallel for collapse(3) reduction(+:total)
    for (ii = 0; ii < params.ny; ii++)
    {
        for (jj = 0; jj < params.nx; jj++)
        {
            for (kk = 0; kk < NSPEEDS; kk++)
            {
                total += cells[ii * params.nx + jj].speeds[kk];
            }
        }
    }
    
    return total;
}

int write_values(const t_param params, t_speed* cells, int* obstacles, double* av_vels)
{
    FILE* fp;                     /* file pointer */
    const double c_sq = 1.0 / 3.0; /* sq. of speed of sound */
    double local_density;         /* per grid cell sum of densities */
    double pressure;              /* fluid pressure in grid cell */
    double u_x;                   /* x-component of velocity in grid cell */
    double u_y;                   /* y-component of velocity in grid cell */
    double u;                     /* norm--root of summed squares--of u_x and u_y */
    
    fp = fopen(FINALSTATEFILE, "w");
    
    if (fp == NULL)
    {
        die("could not open file output file", __LINE__, __FILE__);
    }
    
    int ii, jj;
    for (ii = 0; ii < params.ny; ii++)
    {
        for (jj = 0; jj < params.nx; jj++)
        {
            /* an occupied cell */
            if (obstacles[ii * params.nx + jj])
            {
                u_x = u_y = u = 0.0;
                pressure = params.density * c_sq;
            }
            /* no obstacle */
            else
            {
                local_density = 0.0;
                
                int kk;
                for (kk = 0; kk < NSPEEDS; kk++)
                {
                    local_density += cells[ii * params.nx + jj].speeds[kk];
                }
                
                /* compute x velocity component */
                u_x = (cells[ii * params.nx + jj].speeds[1]
                       + cells[ii * params.nx + jj].speeds[5]
                       + cells[ii * params.nx + jj].speeds[8]
                       - (cells[ii * params.nx + jj].speeds[3]
                          + cells[ii * params.nx + jj].speeds[6]
                          + cells[ii * params.nx + jj].speeds[7]))
                / local_density;
                /* compute y velocity component */
                u_y = (cells[ii * params.nx + jj].speeds[2]
                       + cells[ii * params.nx + jj].speeds[5]
                       + cells[ii * params.nx + jj].speeds[6]
                       - (cells[ii * params.nx + jj].speeds[4]
                          + cells[ii * params.nx + jj].speeds[7]
                          + cells[ii * params.nx + jj].speeds[8]))
                / local_density;
                /* compute norm of velocity */
                u = sqrt((u_x * u_x) + (u_y * u_y));
                /* compute pressure */
                pressure = local_density * c_sq;
            }
            
            /* write to file */
            fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", jj, ii, u_x, u_y, u, pressure, obstacles[ii * params.nx + jj]);
        }
    }
    
    fclose(fp);
    
    fp = fopen(AVVELSFILE, "w");
    
    if (fp == NULL)
    {
        die("could not open file output file", __LINE__, __FILE__);
    }
    
    for (ii = 0; ii < params.maxIters; ii++)
    {
        fprintf(fp, "%d:\t%.12E\n", ii, av_vels[ii]);
    }
    
    fclose(fp);
    
    return EXIT_SUCCESS;
}

void die(const char* message, const int line, const char* file)
{
    fprintf(stderr, "Error at line %d of file %s:\n", line, file);
    fprintf(stderr, "%s\n", message);
    fflush(stderr);
    exit(EXIT_FAILURE);
}

void usage(const char* exe)
{
    fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
    exit(EXIT_FAILURE);
}

