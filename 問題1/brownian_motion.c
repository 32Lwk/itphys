/*
 * brownian_motion.c
 * 
 * 目的: 2次元ブラウン運動を数値的にシミュレートするプログラム
 * 
 * 物理モデル:
 * - ランジュバン方程式に基づくブラウン運動のシミュレーション
 * - 速度の時間発展: dv/dt = -(γ/m)v + sqrt(2γkBT/m) * η(t)
 * - 位置の時間発展: dr/dt = v
 * 
 * 処理の流れ:
 * 1. コマンドライン引数から物理パラメータ（温度T、質量m、摩擦係数γ、時間刻みdt、ステップ数）を取得
 * 2. 初期条件（位置(0,0)、速度(0,0)）を設定
 * 3. オイラー法で時間発展を計算し、各時刻の位置と速度を出力
 */

#include <stdio.h>   // printf関数を使用するため
#include <stdlib.h>  // rand, srand, atof, atoi関数を使用するため
#include <math.h>    // sqrt関数を使用するため
#include <time.h>    // time関数を使用するため

/**
 * Box-Muller変換を用いて標準正規分布N(0,1)に従う乱数を生成する関数
 * 
 * @return 標準正規分布N(0,1)に従う乱数値
 */
double normal_rand() {
    double u1, u2;  // 一様乱数u1, u2を格納する変数
    
    // 0 < u1 < 1 の範囲の一様乱数を生成
    u1 = (rand() + 1.0) / (RAND_MAX + 2.0);
    // 0 < u2 < 1 の範囲の一様乱数を生成
    u2 = (rand() + 1.0) / (RAND_MAX + 2.0);
    
    // Box-Muller変換で正規分布乱数を生成
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

/**
 * メイン関数
 * ランジュバン方程式に基づいて2次元ブラウン運動をシミュレート
 * 
 * @param argc コマンドライン引数の個数
 * @param argv コマンドライン引数の配列
 *             argv[1]: 温度T（デフォルト: 1.0）
 *             argv[2]: 質量m（デフォルト: 1.0）
 *             argv[3]: 摩擦係数γ（デフォルト: 1.0）
 *             argv[4]: 時間刻みdt（デフォルト: 0.01）
 *             argv[5]: ステップ数n_steps（デフォルト: 1000）
 * @return 正常終了時は0を返す
 */
int main(int argc, char *argv[]) {
    // 物理パラメータのデフォルト値を設定
    double gamma = 1.0;  // 摩擦係数
    double kB = 1.0;     // ボルツマン定数（単位系により1.0に設定）
    double T = 1.0;      // 温度
    double m = 1.0;      // 粒子の質量
    double dt = 0.01;    // 時間刻み
    int n_steps = 1000;  // 時間ステップ数
    
    // コマンドライン引数からパラメータを取得（指定されていない場合はデフォルト値を使用）
    if (argc >= 2) T = atof(argv[1]);        // 温度Tを取得
    if (argc >= 3) m = atof(argv[2]);        // 質量mを取得
    if (argc >= 4) gamma = atof(argv[3]);    // 摩擦係数γを取得
    if (argc >= 5) dt = atof(argv[4]);        // 時間刻みdtを取得
    if (argc >= 6) n_steps = atoi(argv[5]);  // ステップ数n_stepsを取得
    
    // 初期条件を設定
    double t = 0.0;   // 時刻（初期値: 0.0）
    double rx = 0.0;  // x座標（初期値: 0.0）
    double ry = 0.0;  // y座標（初期値: 0.0）
    double vx = 0.0;  // x方向の速度（初期値: 0.0）
    double vy = 0.0;  // y方向の速度（初期値: 0.0）
    
    // ランジュバン方程式の係数を事前計算
    double coeff1 = gamma / m;  // 減衰項の係数: -(γ/m)
    // ノイズ項の係数: sqrt(2γkBT/m)（揺動散逸定理から決定）
    double coeff2 = sqrt(2.0 * gamma * kB * T / m);
    
    // 乱数生成器を現在時刻で初期化
    srand((unsigned int)time(NULL));
    
    // 出力ファイルのヘッダー行を出力
    printf("# t x y vx vy\n");
    // 初期状態を出力
    printf("%.15e %.15e %.15e %.15e %.15e\n", t, rx, ry, vx, vy);
    
    // 時間発展のループ（オイラー法で数値積分）
    for (int n = 0; n < n_steps; n++) {
        // 標準正規分布に従う乱数（ホワイトノイズ）を生成
        double eta_x = normal_rand();  // x方向のノイズ
        double eta_y = normal_rand();   // y方向のノイズ
        
        // ランジュバン方程式に基づいて速度を更新
        // dv/dt = -(γ/m)v + sqrt(2γkBT/m) * η(t) をオイラー法で離散化
        vx = vx - coeff1 * vx * dt + coeff2 * sqrt(dt) * eta_x;
        vy = vy - coeff1 * vy * dt + coeff2 * sqrt(dt) * eta_y;
        
        // 位置を更新（dr/dt = v をオイラー法で離散化）
        rx += vx * dt;  // x座標を更新
        ry += vy * dt;  // y座標を更新
        
        // 時刻を更新
        t += dt;
        
        // 現在の状態（時刻、位置、速度）を出力
        printf("%.15e %.15e %.15e %.15e %.15e\n", t, rx, ry, vx, vy);
    }
    
    return 0;  // 正常終了
}
