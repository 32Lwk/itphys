#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

double normal_rand() {
    double u1, u2;
    u1 = (rand() + 1.0) / (RAND_MAX + 2.0);
    u2 = (rand() + 1.0) / (RAND_MAX + 2.0);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

int main(int argc, char *argv[]) {
    int n_samples = atoi(argv[1]);
    srand((unsigned int)time(NULL));
    for (int i = 0; i < n_samples; i++) {
        printf("%.15e\n", normal_rand());
    }
    return 0;
}
