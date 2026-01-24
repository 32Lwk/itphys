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
    double gamma = 1.0, kB = 1.0, T = 1.0, m = 1.0, dt = 0.01;
    int n_steps = 1000;
    if (argc >= 2) T = atof(argv[1]);
    if (argc >= 3) m = atof(argv[2]);
    if (argc >= 4) gamma = atof(argv[3]);
    if (argc >= 5) dt = atof(argv[4]);
    if (argc >= 6) n_steps = atoi(argv[5]);
    
    double t = 0.0, rx = 0.0, ry = 0.0, vx = 0.0, vy = 0.0;
    double coeff1 = gamma / m;
    double coeff2 = sqrt(2.0 * gamma * kB * T / m);
    
    srand((unsigned int)time(NULL));
    printf("# t x y vx vy\n");
    printf("%.15e %.15e %.15e %.15e %.15e\n", t, rx, ry, vx, vy);
    
    for (int n = 0; n < n_steps; n++) {
        double eta_x = normal_rand();
        double eta_y = normal_rand();
        vx = vx - coeff1 * vx * dt + coeff2 * sqrt(dt) * eta_x;
        vy = vy - coeff1 * vy * dt + coeff2 * sqrt(dt) * eta_y;
        rx += vx * dt;
        ry += vy * dt;
        t += dt;
        printf("%.15e %.15e %.15e %.15e %.15e\n", t, rx, ry, vx, vy);
    }
    return 0;
}
