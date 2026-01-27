"""
visualize_trajectories_pure.py

目的: ブラウン運動の2次元軌道を可視化（Pythonのみ版）
- 複数回のシミュレーション実行結果を重ねて表示
- 各軌道の開始点と終了点をマーカーで表示

処理の流れ:
1. コマンドライン引数から実行回数を取得（デフォルト: 5回）
2. 各実行についてブラウン運動をシミュレート
3. 全軌道を2次元平面上に描画
4. 開始点（○）と終了点（□）をマーカーで表示
5. 図をファイルに保存
"""

import numpy as np          # 数値計算ライブラリ
import matplotlib.pyplot as plt  # グラフ描画ライブラリ
import os                   # ファイル操作用
import sys                  # コマンドライン引数の取得用

# 日本語フォントの設定
plt.rcParams['font.family'] = 'Hiragino Sans'
# マイナス記号の文字化けを防ぐ設定
plt.rcParams['axes.unicode_minus'] = False

def simulate_brownian_motion(T=1.0, m=1.0, gamma=1.0, kB=1.0, dt=0.01, n_steps=1000, seed=None):
    """
    ブラウン運動を数値的にシミュレートする関数
    
    @param T: 温度（デフォルト: 1.0）
    @param m: 粒子の質量（デフォルト: 1.0）
    @param gamma: 摩擦係数（デフォルト: 1.0）
    @param kB: ボルツマン定数（デフォルト: 1.0）
    @param dt: 時間刻み（デフォルト: 0.01）
    @param n_steps: 時間ステップ数（デフォルト: 1000）
    @param seed: 乱数のシード（デフォルト: None）
    @return: (t, x, y, vx, vy) のタプル（時刻、x座標、y座標、x速度、y速度の配列）
    """
    # 乱数のシードを設定（再現性のため）
    if seed is not None:
        np.random.seed(seed)
    
    # 初期条件
    rx, ry = 0.0, 0.0  # 初期位置（原点）
    vx, vy = 0.0, 0.0  # 初期速度（ゼロ）
    
    # ランジュバン方程式の係数を事前計算
    coeff1 = gamma / m  # 減衰項の係数: -(γ/m)
    coeff2 = np.sqrt(2.0 * gamma * kB * T / m)  # ノイズ項の係数: sqrt(2γkBT/m)
    
    # 配列を初期化
    t = np.zeros(n_steps + 1)  # 時刻の配列
    x = np.zeros(n_steps + 1)  # x座標の配列
    y = np.zeros(n_steps + 1)  # y座標の配列
    vx_arr = np.zeros(n_steps + 1)  # x速度の配列
    vy_arr = np.zeros(n_steps + 1)  # y速度の配列
    
    # 初期値を設定
    t[0] = 0.0
    x[0] = rx
    y[0] = ry
    vx_arr[0] = vx
    vy_arr[0] = vy
    
    # 時間発展のループ（オイラー法で数値積分）
    for n in range(n_steps):
        # 標準正規分布に従う乱数（ホワイトノイズ）を生成
        eta_x = np.random.normal(0, 1)  # x方向のノイズ
        eta_y = np.random.normal(0, 1)   # y方向のノイズ
        
        # ランジュバン方程式に基づいて速度を更新
        vx = vx - coeff1 * vx * dt + coeff2 * np.sqrt(dt) * eta_x
        vy = vy - coeff1 * vy * dt + coeff2 * np.sqrt(dt) * eta_y
        
        # 位置を更新
        rx += vx * dt  # x座標を更新
        ry += vy * dt  # y座標を更新
        
        # 配列に値を保存
        t[n+1] = (n+1) * dt
        x[n+1] = rx
        y[n+1] = ry
        vx_arr[n+1] = vx
        vy_arr[n+1] = vy
    
    return t, x, y, vx_arr, vy_arr

# スクリプトのディレクトリに移動
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

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
    
    # ブラウン運動をシミュレート（各実行で異なるシードを使用）
    t, x, y, vx, vy = simulate_brownian_motion(seed=i)
    
    # データをdatファイルに保存（Cプログラムと同じ形式: t x y vx vy）
    data = np.column_stack([t, x, y, vx, vy])
    np.savetxt(output_file, data, fmt='%.15e', header='t x y vx vy', comments='#')
    
    # datファイルからデータを読み込む（#で始まるコメント行は無視）
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
plt.savefig(os.path.join('figures', 'trajectories_2d_pure.png'), dpi=150)
# メモリを解放するために図を閉じる
plt.close()
