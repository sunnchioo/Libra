#include <stdio.h>
#include <stdlib.h>

double sum(double &a, double &b) {
    double c = a + b;
    return c;
}

int main() {
    double a = 1.0;
    double b = 2.0;

    double c = sum(a, b);

    printf("sum: %f\n", c);

    return 0;
}