#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include <float.h>
#include <pthread.h>

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

// Global variables
double **points;
double **medoids;
int *clusters;
double *partial_costs;
int num_points;
int num_clusters;
int num_threads;
int dim;

// Function for initial cluster assignment
void* initial_cluster_assignment(void* arg) {
    int thread_id = *((int*)arg);
    int chunk_size = (num_points + num_threads - 1) / num_threads;
    int start = thread_id * chunk_size;
    int end = (start + chunk_size < num_points) ? start + chunk_size : num_points;

    for (int i = start; i < end; i++) {
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
    pthread_exit(NULL);
}

// Function for cluster assignment in each iteration
void* cluster_assignment(void* arg) {
    // Same as initial_cluster_assignment
    int thread_id = *((int*)arg);
    int chunk_size = (num_points + num_threads - 1) / num_threads;
    int start = thread_id * chunk_size;
    int end = (start + chunk_size < num_points) ? start + chunk_size : num_points;

    for (int i = start; i < end; i++) {
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
    pthread_exit(NULL);
}

// Function for finding new medoids
void* find_new_medoids(void* arg) {
    int thread_id = *((int*)arg);
    int chunk_size = (num_clusters + num_threads -1) / num_threads;
    int start = thread_id * chunk_size;
    int end = (start + chunk_size < num_clusters) ? start + chunk_size : num_clusters;

    for (int i = start; i < end; i++) {
        double min_total_distance = DBL_MAX;
        int best_medoid_id = i;
        double *best_medoid = (double *)malloc(dim * sizeof(double));
        if (best_medoid == NULL) {
            fprintf(stderr, "Memory allocation failed for best_medoid\n");
            exit(1);
        }

        for (int j = 0; j < num_points; j++) {
            if (clusters[j] != i) {
                continue;
            }

            double total_distance = 0.0;
            for (int l = 0; l < num_points; l++) {
                if (clusters[l] == i) {
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
        free(best_medoid);
    }

    pthread_exit(NULL);
}

// Function to compute partial costs
void* compute_partial_cost(void* arg) {
    int thread_id = *((int*)arg);
    double local_cost = 0.0;

    int chunk_size = (num_points + num_threads - 1) / num_threads;
    int start = thread_id * chunk_size;
    int end = (start + chunk_size < num_points) ? start + chunk_size : num_points;

    for (int i = start; i < end; i++) {
        for (int j = 0; j < num_clusters; j++) {
            local_cost += euclidean_distance(points[i], medoids[j], dim);
        }
    }

    partial_costs[thread_id] = local_cost;

    pthread_exit(NULL);
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <input_file> <num_clusters> <num_threads>\n", argv[0]);
        return 1;
    }

    FILE *input = fopen(argv[1], "r");
    if (input == NULL) {
        perror("Failed to open input file");
        return 1;
    }

    FILE *clusters_file = fopen("clusters.txt", "w+");
    FILE *medoids_file = fopen("medoids.txt", "w+");
    num_clusters = atoi(argv[2]);
    num_threads = atoi(argv[3]);
    int num_points;
    int dim;

    double start_time = monotonic_seconds();

    // Get number of points and their dimensionality
    char buffer[64];
    fgets(buffer, sizeof(buffer), input);
    buffer[strcspn(buffer, "\n")] = 0;
    num_points = atoi(strtok(buffer, " "));
    dim = atoi(strtok(NULL, " "));

    // Dynamically allocate arrays for storing data
    points = (double **)malloc(num_points * sizeof(double *));
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

    clusters = (int *)malloc(num_points * sizeof(int));
    if (clusters == NULL) {
        fprintf(stderr, "Memory allocation failed for clusters\n");
        return 1;
    }

    medoids = (double **)malloc(num_clusters * sizeof(double *));
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

    // Initialize medoids
    for (int i = 0; i < num_clusters; i++){
        for (int j = 0; j < dim; j++){
            medoids[i][j] = points[i][j];
        }
    }

    // Assign each point to a cluster using Pthreads
    pthread_t threads[num_threads];
    int thread_ids[num_threads];

    for (int t = 0; t < num_threads; t++) {
        thread_ids[t] = t;
        int rc = pthread_create(&threads[t], NULL, initial_cluster_assignment, (void*)&thread_ids[t]);
        if (rc) {
            fprintf(stderr, "Error: unable to create thread, %d\n", rc);
            exit(-1);
        }
    }

    // Wait for all threads to complete
    for (int t = 0; t < num_threads; t++) {
        pthread_join(threads[t], NULL);
    }

    // Run the k-medoids algorithm
    int previous_cost = INT_MAX;
    int current_cost = 0;
    for (int curr_iter = 0; curr_iter < MAX_ITER; curr_iter++){
        // Assign clusters
        for (int t = 0; t < num_threads; t++) {
            thread_ids[t] = t;
            int rc = pthread_create(&threads[t], NULL, cluster_assignment, (void*)&thread_ids[t]);
            if (rc) {
                fprintf(stderr, "Error: unable to create thread, %d\n", rc);
                exit(-1);
            }
        }

        // Wait for all threads to complete
        for (int t = 0; t < num_threads; t++) {
            pthread_join(threads[t], NULL);
        }

        // Find new medoids
        for (int t = 0; t < num_threads; t++) {
            thread_ids[t] = t;
            int rc = pthread_create(&threads[t], NULL, find_new_medoids, (void*)&thread_ids[t]);
            if (rc) {
                fprintf(stderr, "Error: unable to create thread, %d\n", rc);
                exit(-1);
            }
        }

        // Wait for all threads to complete
        for (int t = 0; t < num_threads; t++) {
            pthread_join(threads[t], NULL);
        }

        // Update clusters again (if needed)
        for (int t = 0; t < num_threads; t++) {
            thread_ids[t] = t;
            int rc = pthread_create(&threads[t], NULL, cluster_assignment, (void*)&thread_ids[t]);
            if (rc) {
                fprintf(stderr, "Error: unable to create thread, %d\n", rc);
                exit(-1);
            }
        }

        // Wait for all threads to complete
        for (int t = 0; t < num_threads; t++) {
            pthread_join(threads[t], NULL);
        }

        // Compute total cost
        partial_costs = (double *)malloc(num_threads * sizeof(double));
        if (partial_costs == NULL) {
            fprintf(stderr, "Memory allocation failed for partial_costs\n");
            return 1;
        }

        for (int t = 0; t < num_threads; t++) {
            thread_ids[t] = t;
            int rc = pthread_create(&threads[t], NULL, compute_partial_cost, (void*)&thread_ids[t]);
            if (rc) {
                fprintf(stderr, "Error: unable to create thread, %d\n", rc);
                exit(-1);
            }
        }

        // Wait for all threads to complete
        for (int t = 0; t < num_threads; t++) {
            pthread_join(threads[t], NULL);
        }

        // Sum up partial costs
        current_cost = 0.0;
        for (int t = 0; t < num_threads; t++) {
            current_cost += partial_costs[t];
        }

        free(partial_costs);
        partial_costs = NULL;

        // Check for convergence
        if (abs(previous_cost - current_cost) < 1e-4) {
            break;
        }

        previous_cost = current_cost;
    }

    double end_time = monotonic_seconds();
    print_time(end_time - start_time);

    // Free allocated arrays
    free(clusters);

    for (int i = 0; i < num_clusters; i++) {
        free(medoids[i]);
    }
    free(medoids);

    for (int i = 0; i < num_points; i++) {
        free(points[i]);
    }
    free(points);

    fclose(input);
    fclose(clusters_file);
    fclose(medoids_file);

    return 0;
}