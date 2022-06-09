#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "mpi.h"

void swap(int* v1, int* v2)
{
    int temp = *v1;
    *v1 = *v2;
    *v2 = temp;
}

void sortArray(int *arr, int n)
{
    int i, j, min_idx;
    for (i = 0; i < n - 1; i++)
    {
        min_idx = i;
        for (j = i + 1; j < n; j++)
        {
            if (arr[j] < arr[min_idx])
            {
                min_idx = j;
            }
            swap(&arr[min_idx], &arr[i]);
        }
    }
}

void readFile(int *data)
{
    int i = 0  ;
    FILE *file ;
    file = fopen("/shared/dataset.txt", "r");
    if (file)
	{
        while (fscanf(file, "%d", &data[i]) != EOF)
        {
            i++;
        }
        fclose(file);
    }
    else
    {
        printf("Can't Open this file \n");
    }
}

void get_input(int *num_threads, int *num_points, int *num_bars) {
    printf("\nThreads: ");
    scanf("%d", num_threads);

    printf("\nData Size: ");
    scanf("%d", num_points);

    printf("\nBars: ");
    scanf("%d", num_bars);
}

void distribute_bars(int p, int *offset, int num_bars, int *bars_distribution, int *displs) {
	int c;
	for ( c = 0; c < num_bars; c++) {
        if (c % p == 0) {
            *offset = c;
        }
        bars_distribution[c - *offset]++;
    }

    for ( c = 0; c < p - 1; c++) {
        displs[c + 1] = displs[c] + bars_distribution[c];
    }
}


int main(int argc, char *argv[]) {
    double lower;
    double upper;
    double frame_size;
    int i;
    int j;
    int p;
    int rank;
    int offset;
    int *displs;
    int num_bars;
    int num_points;
    int num_threads;
    int *total_bar_count;
    int local_num_bars;
    int *bars_distribution;
    MPI_Status status;
    /* Start up MPI */
    MPI_Init(&argc, &argv);

    /* Find out process rank */
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* Find out number of process */
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    if (rank == 0) {
        get_input(&num_threads, &num_points, &num_bars);

        bars_distribution = malloc(p * sizeof(int));
        displs = malloc(p * sizeof(int));
        total_bar_count = malloc(num_bars * sizeof(int));

        memset(bars_distribution, 0, p * sizeof(int));
        memset(displs, 0, p * sizeof(int));


        offset = 0;
        distribute_bars(p, &offset, num_bars, bars_distribution, displs);

        local_num_bars = bars_distribution[0];
        offset = displs[0];

        for (i = 1; i < p; i++) {
            MPI_Send(&bars_distribution[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(&displs[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(&num_points, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(&num_bars, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(&num_threads, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(&local_num_bars, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(&offset, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(&num_points, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(&num_bars, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(&num_threads, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
    }

    int data[num_points];

    if (rank == 0) {
        readFile(data);
        sortArray(data, num_points);
        frame_size = (double) (data[num_points - 1]) / num_bars;
        for (i = 1; i < p; i++) {
            MPI_Send(&frame_size, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
            MPI_Send(&data[0], num_points, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(&frame_size, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(&data[0], num_points, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
    }

    int local_bar_counts[local_num_bars];
    memset(local_bar_counts, 0, local_num_bars * sizeof(int));

    for (i = 0; i < local_num_bars; i++) {
        lower = frame_size * (offset + i);
        upper = lower + frame_size;
#pragma omp parallel num_threads(num_threads) default(none) shared(i, j, num_points, rank, data, local_bar_counts, lower, upper)
        {
#pragma omp for schedule(static)
            for (j = 0; j < num_points; j++) {
                if (data[j] > lower && data[j] <= upper) {
                    local_bar_counts[i]++;
                }
            }
        }
    }

    MPI_Gatherv(local_bar_counts, local_num_bars, MPI_INT, total_bar_count, bars_distribution, displs, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        for (i = 0; i < num_bars; i++) {
            lower = frame_size * i;
            upper = lower + frame_size;
            printf("The range start with ");
            printf("%lf", lower);
            printf(", end with ");
            printf("%lf", upper);
            printf(" with count ");
            printf("%d\n", total_bar_count[i]);
        }
    }

    /* shutdown MPI */
    MPI_Finalize();

    return 0;
}