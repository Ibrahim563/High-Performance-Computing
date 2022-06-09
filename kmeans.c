// 20180381 Ahmed Khattab
// 20180379 Ziad Amr
// 20180004 Ibrahim Hany
// 20180331 Waleed Kamal

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#define RECORDS 8
#define MAX_ITERATIONS 15

struct point
{
    float x;
    float y;
};

float euclidean(struct point A, struct point B)
{
    return sqrt(pow(B.x - A.x, 2) + pow(B.y - A.y, 2));
}

int main(int argc, char *argv[])
{
    struct point data_points[RECORDS];

    FILE *file = fopen("points.txt", "r");

    for (int i = 0; i < RECORDS; i++)
    {
        fscanf(file, "%f", &data_points[i].x);
        fscanf(file, "%f", &data_points[i].y);
        printf("Data Point %d, (%0.1f, %0.1f)\n", i, data_points[i].x, data_points[i].y);
    }

    fclose(file);

    int K = 3;
    omp_set_num_threads(K);

    int clusters[K][RECORDS];

    struct point centroids[K];

    for (int i = 0; i < K; i++)
    {
        centroids[i] = data_points[i];
        // printf("Centroid %d, (%f, %f)\n", i, centroids[i].x, centroids[i].y);
    }

    for (int q = 0; q < MAX_ITERATIONS; q++)
    {
        for (int i = 0; i < K; i++)
        {
            for (int j = 0; j < RECORDS; j++)
            {
                clusters[i][j] = -1;
            }
        }

        int id, threads_count, points_per_thread, start, end;

#pragma omp parallel private(id, start, end) shared(threads_count, points_per_thread)
        {
            id = omp_get_thread_num();
            threads_count = omp_get_num_threads();

            points_per_thread = RECORDS / threads_count;

            start = points_per_thread * id;
            end = start + points_per_thread;

            if (end < RECORDS)
            {
                end = RECORDS;
                points_per_thread = start - end;
            }

            for (int i = start; i < end; i++)
            {
                float min_distance = MAXFLOAT;
                int min_index = 0;
                for (int j = 0; j < K; j++)
                {
                    float temp = euclidean(data_points[i], centroids[j]);
                    if (temp < min_distance)
                    {
                        min_distance = temp;
                        min_index = j;
                    }
                }

                clusters[min_index][i] = i;
            }
        }

        for (int i = 0; i < K; i++)
        {
            int sumX = 0;
            int sumY = 0;
            int N = 0;
            for (int j = 0; j < RECORDS; j++)
            {
                if (clusters[i][j] != -1)
                {
                    sumX += data_points[clusters[i][j]].x;
                    sumY += data_points[clusters[i][j]].y;
                    N++;
                }
            }

            centroids[i].x = sumX / N;
            centroids[i].y = sumY / N;
        }
    }

    for (int i = 0; i < K; i++)
    {
        printf("Cluster %d. Centroid: (%0.1f, %0.1f)\n", i + 1, centroids[i].x, centroids[i].y);
        for (int j = 0; j < RECORDS; j++)
        {
            if (clusters[i][j] != -1)
                printf("(%0.1f,%0.1f)", data_points[clusters[i][j]].x, data_points[clusters[i][j]].y);
        }
        printf("\n");
    }

    return 0;
}