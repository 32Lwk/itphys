"""
visualize_msd.py

目的: ブラウン運動の平均二乗変位（MSD: Mean Squared Displacement）を可視化
- 複数回のシミュレーション実行結果からMSDを計算
- 各試行のMSDと平均MSDを表示

処理の流れ:
1. コマンドライン引数から実行回数と温度を取得
2. 各実行についてCプログラム（brownian_motion）を実行して軌道データを生成
3. 各軌道からMSD（r²(t) = x²(t) + y²(t)）を計算
4. 全試行の平均MSDを計算
5. 各試行のMSDと平均MSDを時間に対してプロット
6. 図をファイルに保存
"""

import numpy as np          # 数値計算ライブラリ
import matplotlib.pyplot as plt  # グラフ描画ライブラリ
import subprocess           # 外部プログラム実行用
import os                   # ファイル操作用
import sys                  # コマンドライン引数の取得用

# 日本語フォントの設定
plt.rcParams['font.family'] = 'Hiragino Sans'
# マイナス記号の文字化けを防ぐ設定
plt.rcParams['axes.unicode_minus'] = False

# コマンドライン引数から実行回数を取得（指定がない場合はデフォルト値5を使用）
n_runs = int(sys.argv[1]) if len(sys.argv) > 1 else 5
# コマンドライン引数から温度Tを取得（指定がない場合はデフォルト値1.0を使用）
T = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0

# 物理パラメータ（質量、摩擦係数、ボルツマン定数）
m, gamma, kB = 1.0, 1.0, 1.0

# データと図を保存するディレクトリを作成
os.makedirs('data', exist_ok=True)      # データファイル用ディレクトリ
os.makedirs('figures', exist_ok=True)   # 図ファイル用ディレクトリ

# 各実行の軌道データを格納するリスト
trajectories = []

# 指定された回数分、ブラウン運動のシミュレーションを実行
for i in range(n_runs):
    # 各実行の出力ファイル名を決定
    output_file = os.path.join('data', f'trajectory_{i+1}.dat')
    
    # Cプログラム（brownian_motion）を実行して軌道データを生成
    # 引数: 温度T、質量m、摩擦係数γ
    with open(output_file, 'w') as f:
        subprocess.run(['./brownian_motion', str(T), str(m), str(gamma)], stdout=f)
    
    # 生成されたデータファイルを読み込む（#で始まるコメント行は無視）
    data = np.loadtxt(output_file, comments='#')
    # データから時刻t、x座標、y座標を抽出してリストに追加
    trajectories.append((data[:, 0], data[:, 1], data[:, 2]))

# 最初の軌道から時刻の配列を取得（全軌道で同じ時刻を使用）
t = trajectories[0][0]

# 各試行のMSD（平均二乗変位）を計算
# MSD = r²(t) = x²(t) + y²(t)（2次元の場合）
msd_individual = []
for _, x, y in trajectories:
    # 各時刻での位置の2乗（x² + y²）を計算
    msd_individual.append(x**2 + y**2)

# 全試行の平均MSDを計算
# 各時刻について、全試行のMSDの平均値を計算
msd = np.array([np.mean([msd_individual[j][i] for j in range(n_runs)]) 
                for i in range(len(t))])

# 理論値の計算（参考用、今回は表示しない）
D = kB * T / gamma        # 拡散係数（アインシュタインの関係式: D = kBT/γ）
tau = m / gamma           # 緩和時間
# 理論的なMSD（短時間と長時間の両方を含む一般式）
msd_theory = (4.0 * kB * T / gamma) * (t - tau * (1.0 - np.exp(-t / tau)))
# 長時間極限でのMSD（拡散領域: MSD = 4Dt）
msd_diffusion = 4.0 * D * t

# 図を作成（サイズは10x8インチ）
plt.figure(figsize=(10, 8))

# 各試行のMSDを個別に重ねて表示
colors = plt.cm.tab10(np.linspace(0, 1, n_runs))  # 各試行に異なる色を割り当て
for i in range(n_runs):
    # 各試行のMSDを線で描画（線幅1.5、透明度0.6）
    plt.plot(t, msd_individual[i], '-', linewidth=1.5, alpha=0.6, 
             color=colors[i], label=f'試行 {i+1}')

# 平均MSDを太い黒い線で表示
plt.plot(t, msd, 'k-', linewidth=3, label='平均 <r²(t)>', alpha=0.9)

# 軸ラベルとタイトルを設定
plt.xlabel('時間 t')                    # x軸ラベル
plt.ylabel('平均二乗変位 <r²(t)>')     # y軸ラベル
plt.title(f'平均二乗変位 (n={n_runs}回実行, T={T}, m={m}, γ={gamma})')  # タイトル
plt.legend()     # 凡例を表示
plt.grid(True, alpha=0.3)  # グリッドを表示（透明度0.3）

# レイアウトを調整
plt.tight_layout()
# 図をファイルに保存（解像度150dpi）
plt.savefig(os.path.join('figures', 'msd_plot.png'), dpi=150)
# メモリを解放するために図を閉じる
plt.close()
