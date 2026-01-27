/*
 * normal_rand.c
 * 
 * 目的: Box-Muller変換を用いて標準正規分布N(0,1)に従う乱数を生成するプログラム
 * 
 * 処理の流れ:
 * 1. コマンドライン引数から生成する乱数の個数を取得
 * 2. 乱数生成器を現在時刻で初期化
 * 3. 指定された個数分の正規分布乱数を生成して標準出力に出力
 */

#include <stdio.h>   // printf関数を使用するため
#include <stdlib.h>  // rand, srand, atoi関数を使用するため
#include <math.h>    // sqrt, log, cos, M_PIを使用するため
#include <time.h>    // time関数を使用するため

/**
 * Box-Muller変換を用いて標準正規分布N(0,1)に従う乱数を生成する関数
 * 
 * アルゴリズム:
 * - 2つの一様乱数u1, u2から正規分布乱数を生成
 * - Z = sqrt(-2*ln(u1)) * cos(2*π*u2) が標準正規分布に従う
 * 
 * @return 標準正規分布N(0,1)に従う乱数値
 */
double normal_rand() {
    double u1, u2;  // 一様乱数u1, u2を格納する変数
    
    // 0 < u1 < 1 の範囲の一様乱数を生成（0と1を避けるため+1.0と+2.0を使用）
    u1 = (rand() + 1.0) / (RAND_MAX + 2.0);
    // 0 < u2 < 1 の範囲の一様乱数を生成
    u2 = (rand() + 1.0) / (RAND_MAX + 2.0);
    
    // Box-Muller変換: sqrt(-2*ln(u1)) * cos(2*π*u2) で正規分布乱数を生成
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

/**
 * メイン関数
 * コマンドライン引数から生成する乱数の個数を取得し、その個数分の正規分布乱数を生成
 * 
 * @param argc コマンドライン引数の個数
 * @param argv コマンドライン引数の配列（argv[1]に生成する乱数の個数が入る）
 * @return 正常終了時は0を返す
 */
int main(int argc, char *argv[]) {
    // コマンドライン引数から生成する乱数の個数を取得
    int n_samples = atoi(argv[1]);
    
    // 乱数生成器を現在時刻で初期化（毎回異なる乱数列を生成するため）
    srand((unsigned int)time(NULL));
    
    // 指定された個数分の正規分布乱数を生成して標準出力に出力
    for (int i = 0; i < n_samples; i++) {
        printf("%.15e\n", normal_rand());  // 15桁の指数表記で出力
    }
    
    return 0;  // 正常終了
}
