"""
visualize_trajectories.py

目的: ブラウン運動の2次元軌道を可視化
- 複数回のシミュレーション実行結果を重ねて表示
- 各軌道の開始点と終了点をマーカーで表示

処理の流れ:
1. コマンドライン引数から実行回数を取得（デフォルト: 5回）
2. 各実行についてCプログラム（brownian_motion）を実行して軌道データを生成
3. 全軌道を2次元平面上に描画
4. 開始点（○）と終了点（□）をマーカーで表示
5. 図をファイルに保存
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

# データと図を保存するディレクトリを作成
os.makedirs('data', exist_ok=True)      # データファイル用ディレクトリ
os.makedirs('figures', exist_ok=True)  # 図ファイル用ディレクトリ

# 各実行の軌道データを格納するリスト
trajectories = []

# 指定された回数分、ブラウン運動のシミュレーションを実行
for i in range(n_runs):
    # 各実行の出力ファイル名を決定（trajectory_1.dat, trajectory_2.dat, ...）
    output_file = os.path.join('data', f'trajectory_{i+1}.dat')
    
    # Cプログラム（brownian_motion）を実行して軌道データを生成し、ファイルに保存
    with open(output_file, 'w') as f:
        # デフォルトパラメータでブラウン運動をシミュレート
        subprocess.run(['./brownian_motion'], stdout=f)
    
    # 生成されたデータファイルを読み込む（#で始まるコメント行は無視）
    data = np.loadtxt(output_file, comments='#')
    # データから時刻t、x座標、y座標を抽出してリストに追加
    # data[:, 0]: 時刻t, data[:, 1]: x座標, data[:, 2]: y座標
    trajectories.append((data[:, 0], data[:, 1], data[:, 2]))

# 図を作成（サイズは10x10インチ、正方形）
plt.figure(figsize=(10, 10))

# 各軌道に異なる色を割り当てる（tab10カラーマップからn_runs個の色を取得）
colors = plt.cm.tab10(np.linspace(0, 1, n_runs))

# 全軌道のx座標とy座標を結合して、軸の範囲を決定するために使用
all_x = np.concatenate([x for _, x, _ in trajectories])  # 全x座標を結合
all_y = np.concatenate([y for _, _, y in trajectories])  # 全y座標を結合

# 各軌道を描画
for i, (t, x, y) in enumerate(trajectories):
    # 軌道を線で描画（線幅2.0、透明度0.8）
    plt.plot(x, y, '-', linewidth=2.0, alpha=0.8, color=colors[i], label=f'実行 {i+1}')
    
    # 開始点を○マーカーで表示（マーカーサイズ10、黒い枠線）
    plt.plot(x[0], y[0], 'o', markersize=10, color=colors[i], 
             markeredgecolor='black', markeredgewidth=1)
    
    # 終了点を□マーカーで表示（マーカーサイズ10、黒い枠線）
    plt.plot(x[-1], y[-1], 's', markersize=10, color=colors[i], 
             markeredgecolor='black', markeredgewidth=1)

# 軸ラベルとタイトルを設定
plt.xlabel('x')  # x軸ラベル
plt.ylabel('y')  # y軸ラベル
plt.title(f'ブラウン運動の2次元軌道 ({n_runs}回実行)')  # タイトル
plt.legend()     # 凡例を表示
plt.grid(True, alpha=0.3)  # グリッドを表示（透明度0.3）

# 軸の範囲を設定（全データの範囲に10%のマージンを追加）
margin = 0.1  # マージンの割合（10%）
plt.xlim(all_x.min() - margin * (all_x.max() - all_x.min()), 
         all_x.max() + margin * (all_x.max() - all_x.min()))
plt.ylim(all_y.min() - margin * (all_y.max() - all_y.min()), 
         all_y.max() + margin * (all_y.max() - all_y.min()))

# レイアウトを調整
plt.tight_layout()
# 図をファイルに保存（解像度150dpi）
plt.savefig(os.path.join('figures', 'trajectories_2d.png'), dpi=150)
# メモリを解放するために図を閉じる
plt.close()
