#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include <float.h>
#include <omp.h>

#define MAX_ITER 20

/* Gives us high-resolution timers. */
#define _POSIX_C_SOURCE 199309L
#include <time.h>

/* OSX timer includes */
#ifdef __MACH__
  #include <mach/mach.h>
  #include <mach/mach_time.h>
#endif

/**
* @brief Return the number of seconds since an unspecified time (e.g., Unix
*        epoch). This is accomplished with a high-resolution monotonic timer,
*        suitable for performance timing.
*
* @return The number of seconds.
*/
static inline double monotonic_seconds()
{
#ifdef __MACH__
  /* OSX */
  static mach_timebase_info_data_t info;
  static double seconds_per_unit;
  if(seconds_per_unit == 0) {
    mach_timebase_info(&info);
    seconds_per_unit = (info.numer / info.denom) / 1e9;
  }
  return seconds_per_unit * mach_absolute_time();
#else
  /* Linux systems */
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
#endif
}

/**
* @brief Output the seconds elapsed while clustering.
*
* @param seconds Seconds spent on k-medoids clustering, excluding IO.
*/
static void print_time(double const seconds)
{
  printf("k-medoids clustering time: %0.04fs\n", seconds);
}

double euclidean_distance(double *point1, double *point2, int dim) {
    double distance = 0.0;
    for (int i = 0; i < dim; i++) {
        distance += pow(point1[i] - point2[i], 2);
    }
    return sqrt(distance);
}

double compute_total_cost(double **points, double **medoids, int num_clusters, int num_points, int dim) {
    double total_cost = 0.0;
    #pragma omp parallel for reduction(+:total_cost)
    for (int i = 0; i < num_points; i++) {
        for (int j = 0; j < num_clusters;j++)
        total_cost += euclidean_distance(points[i], medoids[j], dim);
    }
    return total_cost;
}

int main(int argc, char *argv[]) {
    FILE *input = fopen(argv[1], "r");
    FILE *clusters_file = fopen("clusters.txt", "w+");
    FILE *medoids_file = fopen("medoids.txt", "w+");
    int num_clusters = atoi(argv[2]);
    int num_threads = atoi(argv[3]);
    omp_set_num_threads(num_threads);
    int num_points;
    int dim;

    double start_time = monotonic_seconds();

    // Get number of points and their dimensionality
    char buffer[64];
    fgets(buffer, sizeof(buffer), input);
    buffer[strcspn(buffer, "\n")] = 0;
    num_points = atoi(strtok(buffer, " "));
    dim = atoi(strtok(NULL, " "));

    //dynamically allocate arrays for storing data
    double **points = (double **)malloc(num_points * sizeof(double *));
    if (points == NULL) {
        fprintf(stderr, "Memory allocation failed for points\n");
        return 1;
    }
    for (int i = 0; i < num_points; i++) {
        points[i] = (double *)malloc(dim * sizeof(double));
        if (points[i] == NULL) {
            fprintf(stderr, "Memory allocation failed for points[%d]\n", i);
            return 1;
        }
    }

    int *clusters = (int *)malloc(num_points * sizeof(int));
    if (clusters == NULL) {
        fprintf(stderr, "Memory allocation failed for clusters\n");
        return 1;
    }

    double **medoids = (double **)malloc(num_clusters * sizeof(double *));
    if (medoids == NULL) {
        fprintf(stderr, "Memory allocation failed for medoids\n");
        return 1;
    }

    for (int i = 0; i < num_clusters; i++) {
        medoids[i] = (double *)malloc(dim * sizeof(double));
        if (medoids[i] == NULL) {
            fprintf(stderr, "Memory allocation failed for medoids[%d]\n", i);
            return 1;
        }
    }

    // Process input file
    int line_num = 0;
    size_t line_buffer_size = dim * 20;
    char *line = (char *)malloc(line_buffer_size);
    if (line == NULL) {
        fprintf(stderr, "Memory allocation failed for line buffer\n");
        return 1;
    }
    while(fgets(line, line_buffer_size, input)){
        line[strcspn(line, "\n")] = 0;
        int curr_dim = 0;


        char *token = strtok(line, " ");
        while (token != NULL && curr_dim < dim) {
            points[line_num][curr_dim] = strtod(token, NULL);  // Convert token to double
            curr_dim++;
            token = strtok(NULL, " ");
        }
        

      line_num++;
    }

    if (line_num < num_points) {
        fprintf(stderr, "Warning: Expected %d points but read %d points.\n", num_points, line_num);
    }

    
    // Intialize medoids
    for (int i = 0; i < num_clusters;i++){
      for (int j = 0; j < dim; j++){
        medoids[i][j] = points[i][j];
      }
    }

    // Assign each point to a cluster
    #pragma omp parallel for
    for (int i = 0; i < num_points; i++) {
        double min_distance = DBL_MAX;
        int closest_medoid = -1;
        for (int j = 0; j < num_clusters; j++) {
            double distance = euclidean_distance(points[i], medoids[j], dim);
            if (distance < min_distance) {
                min_distance = distance;
                closest_medoid = j;
            }
        }
        clusters[i] = closest_medoid;
    }
    
    // Run the k-medoids algorithm
    int previous_cost = INT_MAX;
    int current_cost = 0;
    for (int curr_iter = 0; curr_iter < MAX_ITER;curr_iter++){
        // assign clusters
        #pragma omp parallel for
        for (int i = 0; i < num_points; i++) {
            double min_distance = DBL_MAX;
            int closest_medoid = -1;
            for (int j = 0; j < num_clusters; j++) {
                double distance = euclidean_distance(points[i], medoids[j], dim);
                if (distance < min_distance) {
                    min_distance = distance;
                    closest_medoid = j;
                }
            }

            clusters[i] = closest_medoid;
        }
        
        // find new medoids
        #pragma omp parallel for
        for (int i = 0; i < num_clusters; i++) {
            double min_total_distance = DBL_MAX;
            int best_medoid_id = i;
            double *best_medoid = (double *)malloc(dim * sizeof(double));

            for (int j = 0; j < num_points; j++) {
                if (clusters[j] != i) {
                    continue;
                }

                double total_distance = 0.0;
                for (int l = 0; l < num_points; l++) {
                    if (clusters[j] == i) {
                        total_distance += euclidean_distance(points[j], points[l], dim);
                    } 
                }

                if (total_distance < min_total_distance) {
                    min_total_distance = total_distance;
                    best_medoid_id = j;
                    memcpy(best_medoid, points[j], dim * sizeof(double));
                }
            }

            clusters[i] = best_medoid_id;
            memcpy(medoids[i], best_medoid, dim * sizeof(double));
        }

        // update cluster medoids for each point
        for (int i = 0; i < num_points; i++) {
            double min_distance = INT_MAX;
            int closest_medoid = -1;
            for (int j = 0; j < num_clusters; j++) {
                double distance = euclidean_distance(points[i], medoids[j], dim);
                if (distance < min_distance) {
                    min_distance = distance;
                    closest_medoid = j;
                }
            }

            clusters[i] = closest_medoid;
        }

        // Check for convergence
        current_cost = compute_total_cost(points, medoids, num_clusters, num_points, dim);

        if (fabs(previous_cost - current_cost) < 1e-4) {
            break;
        }


        previous_cost = current_cost;

    }

    double end_time = monotonic_seconds();
    print_time(end_time - start_time);


    return 0;
}