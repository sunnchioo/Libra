#include <stdio.h>
#include <stdlib.h>

#define MIN_VAL 0.0
#define MAX_VAL 100.0

void random_real(double* vec, int size) {
    double range = MAX_VAL - MIN_VAL;
    for (int i = 0; i < size; i++) {
        double random_0_1 = (double)rand() / RAND_MAX;
        vec[i] = random_0_1 * range + MIN_VAL;
    }
}

int average_array(double* avg, double* vec, size_t size) {

    double sum = 0.0;
    for (size_t i = 0; i < size; i++) {
        sum += vec[i];
    }

    *avg = sum / (double)size;

    return 0;
}

int main() {

    size_t count = 1 << 4;
    double* data = malloc(count * sizeof(double));

    random_real(data, count);

    double avg;
    average_array(&avg, data, count);

    printf("--- data ---\n");
    for (size_t i = 0; i < count; i++) {
        printf("Element[%zu]: %.2f\n", i, data[i]);
    }

    printf("\n--- result ---\n");
    printf("Average: %.2f\n", avg);

    free(data);

    return 0;
}