#include <stdio.h>
#include <stdlib.h>

#define MIN_VAL -2.0 // Random Lower Bound
#define MAX_VAL 2.0  // Random Upper Bound

void random_real(double* vec, int size) {
    double range = MAX_VAL - MIN_VAL;
    for (int i = 0; i < size; i++) {
        double random_0_1 = (double)rand() / RAND_MAX;
        vec[i] = random_0_1 * range + MIN_VAL;
    }
}

int add(double* res, double* a, double* b, size_t size) {
    for (size_t i = 0; i < size; i++) {
        res[i] = a[i] + b[i];
    }

    return 0;
}

int main() {
    size_t count = 10;

    double* x = malloc(count * sizeof(double));
    double* y = malloc(count * sizeof(double));
    double* result = malloc(count * sizeof(double));

    random_real(x, count);
    random_real(y, count);

    add(result, x, y, count);

    for (size_t i = 0; i < count; i++) {
        printf("%f + %f = %f\n", x[i], y[i], result[i]);
    }

    free(x);
    free(y);
    free(result);

    return 0;
}