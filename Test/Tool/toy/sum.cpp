#include <stdio.h>
#include <stdlib.h>

// void sum(double &a, double &b, double &rtn) {
//     rtn = a + b;
// }

// void mult(double &a, double &b, double &rtn) {
//     rtn = a * b;
// }

// int main() {
//     double a = 1.0, b = 2.0;
//     double c = 0;

//     sum(a, b, c);
//     mult(c, c, c);

//     printf("sum: %f\n", c);

//     return 0;
// }

int main() {
    double a = 1.0, b = 2.0;
    double c = 0;

    c = a + b;

    printf("sum: %f\n", c);

    return 0;
}