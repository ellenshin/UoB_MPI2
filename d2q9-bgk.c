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
    double speeds[NSPEEDS];
} t_speed;

/*
 ** function prototypes
 */

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
               int** obstacles_ptr, double** av_vels_ptr);

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

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
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
    t_speed* cells     = NULL;    /* grid containing fluid densities */
    t_speed* tmp_cells = NULL;    /* scratch space */
    int*     obstacles = NULL;    /* grid indicating which cells are blocked */
    double* av_vels   = NULL;     /* a record of the av. velocity computed for each timestep */
    struct timeval timstr;        /* structure to hold elapsed time */
    struct rusage ru;             /* structure to hold CPU time--system and user */
    double tic, toc;              /* doubleing point numbers to calculate elapsed wallclock time */
    double usrtim;                /* doubleing point number to record elapsed user CPU time */
    double systim;                /* doubleing point number to record elapsed system CPU time */
    
    //    int myrank;              /* the rank of this process */
    //    int left;                /* the rank of the process to the left */
    //    int right;               /* the rank of the process to the right */
    //    int size;                /* number of processes in the communicator */
    
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
    
    //    /* MPI_Init returns once it has started up processes */
    //    MPI_Init( &argc, &argv );
    //
    //    /* size and rank will become ubiquitous */
    //    MPI_Comm_size( MPI_COMM_WORLD, &size );
    //    MPI_Comm_rank( MPI_COMM_WORLD, &myrank );
    //
    //    printf("the size is %d", size);
    /* initialise our data structures and load values from file */
    initialise(paramfile, obstaclefile, &params, &cells, &tmp_cells, &obstacles, &av_vels);
    
    /* iterate for maxIters timesteps */
    gettimeofday(&timstr, NULL);
    tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
    
    const double c_sq = 3.0; /* square of speed of sound */
    const double w0 = 4.0 / 9.0;  /* weighting factor */
    const double w1 = 1.0 / 9.0;  /* weighting factor */
    
    
    const double two_c_sq_c_sq = 4.5;
    const double two_c_sq = 1.5;
    
    const int dimension = params.ny*params.nx;
    double w3 = params.density * params.accel / 9.0;
    double w4 = params.density * params.accel / 36.0;
    
    int tt;
    for (tt = 0; tt < params.maxIters; tt++)
    {
        
        //int index;
        //    double *speed;
        //    t_speed* cell;
        
        
        int    tot_cells = 0;  /* no. of cells used in calculation */
        double tot_u = 0.0;          /* accumulated magnitudes of velocity for each cell */
        /* modify the 2nd row of the grid */
        int ii = params.ny - 2;
        int jj;
        //#pragma omp parallel num_threads(2)
        {
            for (jj = 0; jj < params.nx; jj++)
            {
                {
                    /* if the cell is not occupied and
                     ** we don't send a negative density */
                    if (!obstacles[ii * params.nx + jj]
                        && (cells[ii * params.nx + jj].speeds[3] - w3) > 0.0
                        && (cells[ii * params.nx + jj].speeds[6] - w4) > 0.0
                        && (cells[ii * params.nx + jj].speeds[7] - w4) > 0.0)
                    {
                        //#pragma omp sections
                        
                        //#pragma omp section
                        
                        cells[ii * params.nx + jj].speeds[1] += w3;
                        cells[ii * params.nx + jj].speeds[5] += w4;
                        cells[ii * params.nx + jj].speeds[8] += w4;
                        
                        
                        
                        //#pragma omp section
                        
                        cells[ii * params.nx + jj].speeds[3] -= w3;
                        cells[ii * params.nx + jj].speeds[6] -= w4;
                        cells[ii * params.nx + jj].speeds[7] -= w4;
                        
                        /* decrease 'west-side' densities */
                        
                        /* increase 'east-side' densities */
                        
                    }
                }
            }
            
        }
        
        
        
        //timestep(params, cells, tmp_cells, obstacles);
        
        
        //int chunk = (params.ny*params.nx) / 16;
        /* loop over the cells in the grid
         ** NB the collision step is called after
         ** the propagate step and so values of interest
         ** are in the scratch-space grid */
        
        /* loop over _all_ cells */
        for (ii = 0; ii < params.ny; ii++)
        {
            for (jj = 0; jj < params.nx; jj++)
            {
                
                int index = ii * params.nx + jj;
                double* current_speed = cells[index].speeds;
                /* determine indices of axis-direction neighbours
                 ** respecting periodic boundary conditions (wrap around) */
                int y_n = (ii == params.ny) ? 0 : (ii + 1);
                int x_e = (ii == params.nx) ? 0 : (jj + 1);;
                int y_s = (ii == 0) ? (ii + params.ny - 1) : (ii - 1);
                int x_w = (jj == 0) ? (jj + params.nx - 1) : (jj - 1);
                int y_s_mult = y_s * params.nx;
                int y_n_mult = y_n * params.nx;
                int ii_mult = ii*params.nx;
                /* propagate densities to neighbouring cells, following
                 ** appropriate directions of travel and writing into
                 ** scratch space grid */
                tmp_cells[ii_mult + jj].speeds[0]  = current_speed[0]; /* central cell, no movement */
                tmp_cells[ii_mult + x_e].speeds[1] = current_speed[1]; /* east */
                tmp_cells[y_n_mult + jj].speeds[2]  = current_speed[2]; /* north */
                tmp_cells[ii_mult + x_w].speeds[3] = current_speed[3]; /* west */
                tmp_cells[y_s_mult + jj].speeds[4]  = current_speed[4]; /* south */
                tmp_cells[y_n_mult + x_e].speeds[5] = current_speed[5]; /* north-east */
                tmp_cells[y_n_mult + x_w].speeds[6] = current_speed[6]; /* north-west */
                tmp_cells[y_s_mult + x_w].speeds[7] = current_speed[7]; /* south-west */
                tmp_cells[y_s_mult + x_e].speeds[8] = current_speed[8]; /* south-east */
                
                
            }
        }
        for (ii = 0; ii < dimension; ii++)
        {
            /* don't consider occupied cells */
            
            double* tmp_speed = tmp_cells[ii].speeds;
            double* current_speed = cells[ii].speeds;
            
            double tmp_speed_0 = tmp_speed[0];
            double tmp_speed_1 = tmp_speed[1];
            double tmp_speed_2 = tmp_speed[2];
            double tmp_speed_3 = tmp_speed[3];
            double tmp_speed_4 = tmp_speed[4];
            double tmp_speed_5 = tmp_speed[5];
            double tmp_speed_6 = tmp_speed[6];
            double tmp_speed_7 = tmp_speed[7];
            double tmp_speed_8 = tmp_speed[8];
            
            if (!obstacles[ii])
            {
                /* compute local density total */
                
                //double local_density = 0.0;
                
                
                
                //int kk;
                //                    for (kk = 0; kk < NSPEEDS; kk++)
                //                    {
                //                        local_density += tmp_speed[kk];
                //                    }
                
                //local_density = (tmp_speed[0] + tmp_speed[1] + tmp_speed[2] + tmp_speed[3] + tmp_speed[4] + tmp_speed[5] + tmp_speed[6] + tmp_speed[7] + tmp_speed[8]);
                
                /* compute x velocity component */
                double u_x = (tmp_speed[1]
                              + tmp_speed[5]
                              + tmp_speed[8]
                              - (tmp_speed[3]
                                 + tmp_speed[6]
                                 + tmp_speed[7]))
                / (tmp_speed[0] + tmp_speed[1] + tmp_speed[2] + tmp_speed[3] + tmp_speed[4] + tmp_speed[5] + tmp_speed[6] + tmp_speed[7] + tmp_speed[8]);
                /* compute y velocity component */
                double u_y = (tmp_speed[2]
                              + tmp_speed[5]
                              + tmp_speed[6]
                              - (tmp_speed[4]
                                 + tmp_speed[7]
                                 + tmp_speed[8]))
                / (tmp_speed[0] + tmp_speed[1] + tmp_speed[2] + tmp_speed[3] + tmp_speed[4] + tmp_speed[5] + tmp_speed[6] + tmp_speed[7] + tmp_speed[8]);
                
                double u = u_x*u_x + u_y*u_y;
                double local_density_mult = u * (two_c_sq);
                double w1_loc = (w1 * (tmp_speed[0] + tmp_speed[1] + tmp_speed[2] + tmp_speed[3] + tmp_speed[4] + tmp_speed[5] + tmp_speed[6] + tmp_speed[7] + tmp_speed[8]));
                double w2_loc = w1_loc/4.0;
                
                
                *(current_speed) = tmp_speed_0 + params.omega * ((w0* (tmp_speed[0] + tmp_speed[1] + tmp_speed[2] + tmp_speed[3] + tmp_speed[4] + tmp_speed[5] + tmp_speed[6] + tmp_speed[7] + tmp_speed[8])
                                                                  * (1.0 - local_density_mult)) - tmp_speed_0);
                *(current_speed+1) = tmp_speed_1 + params.omega * ((w1_loc * (1.0 + u_x * c_sq
                                                                              + (u_x * u_x) * (two_c_sq_c_sq)
                                                                              - local_density_mult)) - tmp_speed_1);
                *(current_speed+2) = tmp_speed_2 + params.omega * ((w1_loc * (1.0 + u_y * c_sq
                                                                              + (u_y * u_y) * (two_c_sq_c_sq)
                                                                              - local_density_mult)) - tmp_speed_2);
                *(current_speed+3) = tmp_speed_3 + params.omega * ((w1_loc * (1.0 + (-u_x) * c_sq
                                                                              + ((-u_x) * (-u_x)) * (two_c_sq_c_sq)
                                                                              - local_density_mult)) - tmp_speed_3);
                *(current_speed+4) = tmp_speed_4 + params.omega * ((w1_loc * (1.0 + (-u_y) * c_sq
                                                                              + ((-u_y) * (-u_y)) * (two_c_sq_c_sq)
                                                                              - local_density_mult)) - tmp_speed_4);
                *(current_speed+5) = tmp_speed_5 + params.omega * ((w2_loc * (1.0 + (u_x + u_y) * c_sq
                                                                              + ((u_x + u_y) * (u_x + u_y)) * (two_c_sq_c_sq)
                                                                              - local_density_mult)) - tmp_speed_5);
                *(current_speed+6) = tmp_speed_6 + params.omega * ((w2_loc * (1.0 + (- u_x + u_y) * c_sq
                                                                              + ((- u_x + u_y) * (- u_x + u_y)) * (two_c_sq_c_sq)
                                                                              - local_density_mult)) - tmp_speed_6);
                *(current_speed+7) = tmp_speed_7 + params.omega * ((w2_loc * (1.0 + (- u_x - u_y) * c_sq
                                                                              + ((- u_x - u_y) * (- u_x - u_y)) * (two_c_sq_c_sq)
                                                                              - local_density_mult)) - tmp_speed_7);
                *(current_speed+8) = tmp_speed_8 + params.omega * ((w2_loc * (1.0 + ( u_x - u_y) * c_sq
                                                                              + (( u_x - u_y) * ( u_x - u_y)) * (two_c_sq_c_sq)
                                                                              - local_density_mult)) - tmp_speed_8);                    //local_density = 0.0;
                
                //                    for (kk = 0; kk < NSPEEDS; kk++)
                //                    {
                //                        local_density += current_speed[kk];
                //                    }
                
                //local_density = current_speed[0] + current_speed[1] + current_speed[2] + current_speed[3] + current_speed[4] + current_speed[5] + current_speed[6] + current_speed[7] + current_speed[8];
                
                
                /* accumulate the norm of x- and y- velocity components */
                
                tot_u += sqrt(u);
                /* increase counter of inspected cells */
                //++tot_cells_1;
                
                
            }else {
                /* called after propagate, so taking values from scratch space
                 ** mirroring, and writing into main grid */
                //                    double* current_speed_0 = current_speed;
                //                    double* current_speed_1 = (current_speed+1);
                //                    double* current_speed_2 = (current_speed+2);
                //                    double* current_speed_3 = (current_speed+3);
                //                    double* current_speed_4 = (current_speed+4);
                //                    double* current_speed_5 = (current_speed+5);
                //                    double* current_speed_6 = (current_speed+6);
                //                    double* current_speed_7 = (current_speed+7);
                //                    double* current_speed_8 = (current_speed+8);
                
                *(current_speed+1) = tmp_speed[3];
                *(current_speed+2) = tmp_speed[4];
                *(current_speed+3) = tmp_speed[1];
                *(current_speed+4) = tmp_speed[2];
                *(current_speed+5) = tmp_speed[7];
                *(current_speed+6) = tmp_speed[8];
                *(current_speed+7) = tmp_speed[5];
                *(current_speed+8) = tmp_speed[6];
                ++tot_cells;
                
            }
        }
        av_vels[tt] = tot_u / (double)(params.ny*params.nx-tot_cells);
        
    }
    
    
    
    gettimeofday(&timstr, NULL);
    toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
    getrusage(RUSAGE_SELF, &ru);
    timstr = ru.ru_utime;
    usrtim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
    timstr = ru.ru_stime;
    systim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
    
    for(tt=0; tt<params.maxIters; tt++){
#ifdef DEBUG
        printf("==timestep: %d==\n", tt);
        printf("av velocity: %.12E\n", av_vels[tt]);
        printf("tot density: %.12E\n", total_density(params, cells));
        
#endif
    }
    /* write final values and free memory */
    printf("==done==\n");
    printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, cells, obstacles));
    printf("Elapsed time:\t\t\t%.6lf (s)\n", toc - tic);
    printf("Elapsed user CPU time:\t\t%.6lf (s)\n", usrtim);
    printf("Elapsed system CPU time:\t%.6lf (s)\n", systim);
    write_values(params, cells, obstacles, av_vels);
    finalise(&params, &cells, &tmp_cells, &obstacles, &av_vels);
    
    //MPI_Finalize();
    return EXIT_SUCCESS;
}

double av_velocity(const t_param params, t_speed* cells, int* obstacles)
{
    int    tot_cells = 0;  /* no. of cells used in calculation */
    double tot_u = 0.0;          /* accumulated magnitudes of velocity for each cell */
    
    /* initialise */
    
    int ii;
    for (ii = 0; ii < params.ny*params.nx; ii++)
    {
        /* ignore occupied cells */
        if (!obstacles[ii])
        {
            /* local density total */
            double* current_speed = cells[ii].speeds;
            double u_x = (current_speed[1]
                          + current_speed[5]
                          + current_speed[8]
                          - (current_speed[3]
                             + current_speed[6]
                             + current_speed[7]))
            / (current_speed[0] + current_speed[1] + current_speed[2] + current_speed[3] + current_speed[4] + current_speed[5] + current_speed[6] + current_speed[7] + current_speed[8]);
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
            /* increase counter of inspected cells */
            //++tot_cells;
            /* increase counter of inspected cells */
            
            
        } else
            ++tot_cells;
        
    }
    
    
    return tot_u / (double)(params.ny*params.nx-tot_cells);
}

int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
               int** obstacles_ptr, double** av_vels_ptr)
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
    
    /* main grid */
    *cells_ptr = (t_speed*)malloc(sizeof(t_speed) * (params->ny * params->nx));
    
    if (*cells_ptr == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);
    
    /* 'helper' grid, used as scratch space */
    *tmp_cells_ptr = (t_speed*)malloc(sizeof(t_speed) * (params->ny * params->nx));
    
    if (*tmp_cells_ptr == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);
    
    /* the map of obstacles */
    *obstacles_ptr = malloc(sizeof(int) * (params->ny * params->nx));
    
    if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);
    
    /* initialise densities */
    double w0 = params->density * 4.0 / 9.0;
    double w1 = params->density      / 9.0;
    double w2 = params->density      / 36.0;
    
    int ii;
    int jj;
    for (ii = 0; ii < params->ny; ii++)
    {
        for (jj = 0; jj < params->nx; jj++)
        {
            /* centre */
            (*cells_ptr)[ii * params->nx + jj].speeds[0] = w0;
            /* axis directions */
            (*cells_ptr)[ii * params->nx + jj].speeds[1] = w1;
            (*cells_ptr)[ii * params->nx + jj].speeds[2] = w1;
            (*cells_ptr)[ii * params->nx + jj].speeds[3] = w1;
            (*cells_ptr)[ii * params->nx + jj].speeds[4] = w1;
            /* diagonals */
            (*cells_ptr)[ii * params->nx + jj].speeds[5] = w2;
            (*cells_ptr)[ii * params->nx + jj].speeds[6] = w2;
            (*cells_ptr)[ii * params->nx + jj].speeds[7] = w2;
            (*cells_ptr)[ii * params->nx + jj].speeds[8] = w2;
        }
    }
    
    /* first set all cells in obstacle array to zero */
    for (ii = 0; ii < params->ny; ii++)
    {
        for (jj = 0; jj < params->nx; jj++)
        {
            (*obstacles_ptr)[ii * params->nx + jj] = 0;
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
        (*obstacles_ptr)[yy * params->nx + xx] = blocked;
    }
    
    /* and close the file */
    fclose(fp);
    
    /*
     ** allocate space to hold a record of the avarage velocities computed
     ** at each timestep
     */
    *av_vels_ptr = (double*)malloc(sizeof(double) * params->maxIters);
    
    return EXIT_SUCCESS;
}

int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
             int** obstacles_ptr, double** av_vels_ptr)
{
    /*
     ** free up allocated memory
     */
    free(*cells_ptr);
    *cells_ptr = NULL;
    
    free(*tmp_cells_ptr);
    *tmp_cells_ptr = NULL;
    
    free(*obstacles_ptr);
    *obstacles_ptr = NULL;
    
    free(*av_vels_ptr);
    *av_vels_ptr = NULL;
    
    return EXIT_SUCCESS;
}


double calc_reynolds(const t_param params, t_speed* cells, int* obstacles)
{
    const double viscosity = 1.0 / 6.0 * (2.0 / params.omega - 1.0);
    
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

