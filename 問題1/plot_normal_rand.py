"""
plot_normal_rand.py

目的: 正規分布乱数の生成と可視化
- 異なるサンプル数（50, 100, 1000）で正規分布乱数を生成
- 各サンプル数のヒストグラムを描画し、理論的な正規分布と比較

処理の流れ:
1. Cプログラム（normal_rand）を実行して正規分布乱数を生成
2. 生成されたデータを読み込み
3. ヒストグラムと理論曲線を描画して比較
4. 図をファイルに保存
"""

import numpy as np          # 数値計算ライブラリ
import matplotlib.pyplot as plt  # グラフ描画ライブラリ
import subprocess           # 外部プログラム実行用
import os                   # ファイル操作用

# 日本語フォントの設定（Hiragino Sansを使用）
plt.rcParams['font.family'] = 'Hiragino Sans'
# マイナス記号の文字化けを防ぐ設定
plt.rcParams['axes.unicode_minus'] = False

# データと図を保存するディレクトリを作成（既に存在する場合は何もしない）
os.makedirs('data', exist_ok=True)      # データファイル用ディレクトリ
os.makedirs('figures', exist_ok=True)  # 図ファイル用ディレクトリ

# 生成する乱数のサンプル数のリスト（50, 100, 1000の3通り）
n_samples_list = [50, 100, 1000]
# 各サンプル数に対応するデータを格納するリスト
data_list = []

# 各サンプル数について正規分布乱数を生成
for n_samples in n_samples_list:
    # ファイル名を決定（1000の場合はnormal_rand.dat、それ以外はnormal_rand_{n_samples}.dat）
    filename = f'normal_rand_{n_samples}.dat' if n_samples != 1000 else 'normal_rand.dat'
    # データディレクトリ内のファイルパスを作成
    filepath = os.path.join('data', filename)
    
    # Cプログラム（normal_rand）を実行して正規分布乱数を生成し、ファイルに保存
    with open(filepath, 'w') as f:
        # subprocess.runで外部プログラムを実行し、標準出力をファイルに書き込む
        subprocess.run(['./normal_rand', str(n_samples)], stdout=f)
    
    # 生成されたデータファイルを読み込む
    data = np.loadtxt(filepath)
    # サンプル数とデータのペアをリストに追加
    data_list.append((n_samples, data))

# 3つのサブプロットを持つ図を作成（横に3つ並べる、サイズは18x5インチ）
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 各サンプル数についてヒストグラムと理論曲線を描画
for idx, (n_samples, data) in enumerate(data_list):
    # ヒストグラムを描画（ビン数20、密度正規化、透明度0.7、黒い枠線）
    axes[idx].hist(data, bins=20, density=True, alpha=0.7, edgecolor='black')
    
    # 理論的な正規分布曲線を描画するためのx座標を生成（データの最小値から最大値まで1000点）
    x = np.linspace(data.min(), data.max(), 1000)
    # 標準正規分布N(0,1)の確率密度関数: (1/√(2π)) * exp(-x²/2)
    theoretical = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * x**2)
    
    # 理論曲線を赤い実線で描画
    axes[idx].plot(x, theoretical, 'r-', linewidth=2, label='理論値 N(0,1)')
    
    # 軸ラベルとタイトルを設定
    axes[idx].set_xlabel('値')              # x軸ラベル
    axes[idx].set_ylabel('確率密度')        # y軸ラベル
    axes[idx].set_title(f'正規分布乱数のヒストグラム (n={n_samples})')  # タイトル
    axes[idx].legend()                      # 凡例を表示
    axes[idx].grid(True, alpha=0.3)         # グリッドを表示（透明度0.3）

# レイアウトを調整して重なりを防ぐ
plt.tight_layout()
# 図をファイルに保存（解像度150dpi）
plt.savefig(os.path.join('figures', 'normal_rand_hist_all.png'), dpi=150)
# メモリを解放するために図を閉じる
plt.close()
