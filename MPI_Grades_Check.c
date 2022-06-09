#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#define RECORD_LENGTH 13
#define NUM_OF_RECORDS 20

int main(int argc, char **argv)
{
    int my_rank;
    int p;
    int tag = 0;
    MPI_Status status;

    char students[NUM_OF_RECORDS][RECORD_LENGTH];
    int passed_per_core;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    int record_per_core = NUM_OF_RECORDS / (p - 1);
    int remainder;

    if (my_rank == 0)
    {
        FILE *file = fopen("students.txt", "r");

        for (int i = 0; i < NUM_OF_RECORDS; i++)
            fgets(students[i], RECORD_LENGTH, file);

        for (int i = 1; i < p; i++)
        {
            if (i == (p - 1))
                remainder = NUM_OF_RECORDS % (p - 1);
            else
                remainder = 0;

            MPI_Send(&remainder, 1, MPI_INT, i, tag, MPI_COMM_WORLD);
            MPI_Send(&students[(i - 1) * (record_per_core)], RECORD_LENGTH * (record_per_core + remainder), MPI_CHAR, i, 0, MPI_COMM_WORLD);
        }

        fclose(file);
    }
    else
    {
        MPI_Recv(&remainder, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
        MPI_Recv(&students, RECORD_LENGTH * (record_per_core + remainder), MPI_CHAR, 0, tag, MPI_COMM_WORLD, &status);
        passed_per_core = 0;

        for (int i = 0; i < record_per_core + remainder; i++)
        {
            char *id = strtok(students[i], " ");
            int grade = atoi(strtok(NULL, " "));

            if (grade >= 50)
            {
                printf("Core %d: %s has passed the exam.\n", my_rank, id);
                passed_per_core++;
            }
            else
                printf("Core %d, %s has failed. Please repeat the exam.\n", my_rank, id);
        }

        MPI_Send(&passed_per_core, 1, MPI_INT, 0, tag, MPI_COMM_WORLD);
    }

    if (my_rank == 0)
    {
        int passed = 0;
        for (int i = 1; i < p; i++)
        {
            MPI_Recv(&passed_per_core, 1, MPI_INT, i, tag, MPI_COMM_WORLD, &status);
            passed += passed_per_core;
        }

        printf("Total number of students that passed the exam is %d out of %d\n", passed, NUM_OF_RECORDS);
    }

    MPI_Finalize();
    return 0;
}
