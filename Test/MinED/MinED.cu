#include <stdio.h>
#include <stdlib.h>

#include "FlyHE.h"

int main() {
    FlyHE.initSIMD();
    FlyHE.initSISD();
    FlyHE.initConversion();

    size_t input_len = 8;
    double a[input_len], b[input_len];
    double min = 0;

    for (int i = 0; i < 8; i++) {
        scanf("%lf %lf", &a[i], &b[i]);
    }

    FlyHESIMDCipher simd_ct_a = FlyHE.Encrypt(a, input_len);
    FlyHESIMDCipher simd_ct_b = FlyHE.Encrypt(b, input_len);
    FlyHESIMDCipher simd_ct_c;

    simd_ct_c = FlyHE.SIMDSub(simd_ct_a, simd_ct_b);
    simd_ct_c = FlyHE.SIMDMult(simd_ct_c, simd_ct_c);

    FlyHESISDCipher sisd_ct_c = FlyHE.SIMDToSISD(simd_ct_c);

    FlyHESISDCipher sisd_ct_min = FlyHE.SISDMin(sisd_ct_c);

    double min = FlyHE.SISDDecrypt(sisd_ct_min);

    printf("min: %f\n", min);

    return 0;
}