/*
 * report1_haruki.c
 *
 * 問題1用のCプログラム統合版
 *
 * 1. 正規分布乱数（Box-Muller）の生成
 * 2. 2次元ブラウン運動（ランジュバン方程式）のシミュレーション
 *
 * 使い方:
 *   正規分布乱数を n 個生成:     ./report1_haruki normal_rand <n>
 *   ブラウン運動をシミュレート:  ./report1_haruki [T] [m] [gamma] [dt] [n_steps]
 *     省略時: T=1.0, m=1.0, gamma=1.0, dt=0.01, n_steps=1000
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

/**
 * Box-Muller変換を用いて標準正規分布N(0,1)に従う乱数を生成する関数
 */
double normal_rand(void) {
    double u1 = (rand() + 1.0) / (RAND_MAX + 2.0);
    double u2 = (rand() + 1.0) / (RAND_MAX + 2.0);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

/**
 * 正規分布乱数モード: コマンドライン第2引数で指定した個数だけ乱数を標準出力に出力
 */
static int run_normal_rand(int n_samples) {
    srand((unsigned int)time(NULL));
    for (int i = 0; i < n_samples; i++) {
        printf("%.15e\n", normal_rand());
    }
    return 0;
}

/**
 * ブラウン運動モード: ランジュバン方程式に基づく2次元シミュレーション
 * 出力形式: # t x y vx vy のヘッダー付きで、各行に t x y vx vy を出力
 */
static int run_brownian_motion(double T, double m, double gamma,
                               double dt, int n_steps) {
    const double kB = 1.0;
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

int main(int argc, char *argv[]) {
    if (argc >= 2 && strcmp(argv[1], "normal_rand") == 0) {
        /* 正規分布乱数モード */
        int n_samples = (argc >= 3) ? atoi(argv[2]) : 1000;
        return run_normal_rand(n_samples);
    }

    /* ブラウン運動モード: デフォルト T=1.0, m=1.0, gamma=1.0, dt=0.01, n_steps=1000 */
    double T = (argc >= 2) ? atof(argv[1]) : 1.0;
    double m = (argc >= 3) ? atof(argv[2]) : 1.0;
    double gamma = (argc >= 4) ? atof(argv[3]) : 1.0;
    double dt = (argc >= 5) ? atof(argv[4]) : 0.01;
    int n_steps = (argc >= 6) ? atoi(argv[5]) : 1000;

    return run_brownian_motion(T, m, gamma, dt, n_steps);
}
