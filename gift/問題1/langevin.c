/*
 * 問題1 (2): ランジュバン方程式に基づく2次元軌道の計算（C言語）
 *
 * m dv/dt = -γv + ξ(t),  dr/dt = v
 * 初期条件: r0=(0,0), v0=(0,0)
 * 1000 ステップ, Δt=0.01, γ=kB=T=m=1
 * Box-Muller で正規乱数生成
 * 出力: t, x, y, vx, vy を標準出力へ
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Box-Muller: 平均0, 分散1の正規乱数 */
double normal_rand(void) {
    double u1, u2;
    u1 = (rand() + 1.0) / (RAND_MAX + 2.0);
    u2 = (rand() + 1.0) / (RAND_MAX + 2.0);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

int main(int argc, char **argv) {
    const double gamma = 1.0, kB = 1.0, T = 1.0, m = 1.0;
    const double dt = 0.01;
    const int n_steps = 1000;

    double coeff1 = gamma / m;
    double coeff2 = sqrt(2.0 * gamma * kB * T / m) * sqrt(dt);

    double x = 0.0, y = 0.0;
    double vx = 0.0, vy = 0.0;
    int seed = (argc > 1) ? atoi(argv[1]) : 0;
    if (seed != 0) srand((unsigned)seed);

    /* ヘッダ行 */
    printf("# t x y vx vy\n");
    printf("%.6f %.6f %.6f %.6f %.6f\n", 0.0, x, y, vx, vy);

    for (int n = 0; n < n_steps; n++) {
        double eta_x = normal_rand();
        double eta_y = normal_rand();
        vx = vx - coeff1 * vx * dt + coeff2 * eta_x;
        vy = vy - coeff1 * vy * dt + coeff2 * eta_y;
        x += vx * dt;
        y += vy * dt;
        double t = (n + 1) * dt;
        printf("%.6f %.6f %.6f %.6f %.6f\n", t, x, y, vx, vy);
    }
    return 0;
}
