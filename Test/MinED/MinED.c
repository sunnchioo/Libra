#include <stdio.h>
#include <stdlib.h>

int main() {
    double a[8], b[8], c[8];
    double min = 0;

    for (int i = 0; i < 8; i++) {
        scanf("%lf %lf", &a[i], &b[i]);
    }

    for (int i = 0; i < 8; i++) {
        c[i] = a[i] - b[i];
        c[i] = c[i] * c[i];
    }

    for (int i = 0; i < 8; i++) {
        if (min > c[i]) {
            min = c[i];
        }
    }

    printf("min: %f\n", min);

    return 0;
}