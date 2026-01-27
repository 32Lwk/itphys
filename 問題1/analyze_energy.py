"""
analyze_energy.py

目的: ブラウン運動する粒子の運動エネルギー分布関数を解析
- 異なる温度での運動エネルギー分布を計算
- 理論的なボルツマン分布と比較

処理の流れ:
1. 複数の温度（0.5, 1.0, 2.0）についてシミュレーションを実行
2. 各温度について複数回実行して統計を取る
3. 各時刻での運動エネルギー E = (1/2) * m * (vx² + vy²) を計算
4. エネルギー分布のヒストグラムを作成
5. 理論的なボルツマン分布 P(E) = (1/kBT) * exp(-E/kBT) と比較
6. 図をファイルに保存
"""

import numpy as np          # 数値計算ライブラリ
import matplotlib.pyplot as plt  # グラフ描画ライブラリ
import subprocess           # 外部プログラム実行用
import os                   # ファイル操作用
import sys                  # コマンドライン引数の取得用（今回は未使用）

# 日本語フォントの設定
plt.rcParams['font.family'] = 'Hiragino Sans'
# マイナス記号の文字化けを防ぐ設定
plt.rcParams['axes.unicode_minus'] = False

# 解析する温度のリスト（3通り）
T_values = [0.5, 1.0, 2.0]

# 物理パラメータ
m = 1.0      # 粒子の質量
gamma = 1.0  # 摩擦係数
kB = 1.0     # ボルツマン定数（単位系により1.0に設定）
dt = 0.01    # 時間刻み
n_steps = 1000  # 時間ステップ数

# データと図を保存するディレクトリを作成
os.makedirs('data', exist_ok=True)      # データファイル用ディレクトリ
os.makedirs('figures', exist_ok=True)   # 図ファイル用ディレクトリ

# 各温度についてのエネルギー分布を格納するリスト
energies_by_T = []

# 各温度についてシミュレーションを実行
for T in T_values:
    # この温度での全エネルギー値を格納するリスト
    energies = []
    # 統計を取るために複数回実行（より良い分布を得るため）
    n_runs = 10
    
    # 各実行について
    for run in range(n_runs):
        # 出力ファイル名を決定（温度と実行回数を含む）
        output_file = os.path.join('data', f'energy_T{T}_run{run+1}.dat')
        
        # Cプログラム（brownian_motion）を実行して軌道データを生成
        # 引数: 温度T、質量m、摩擦係数γ、時間刻みdt、ステップ数n_steps
        with open(output_file, 'w') as f:
            subprocess.run(['./brownian_motion', str(T), str(m), str(gamma), 
                          str(dt), str(n_steps)], stdout=f)
        
        # 生成されたデータファイルを読み込む（#で始まるコメント行は無視）
        data = np.loadtxt(output_file, comments='#')
        t = data[:, 0]   # 時刻
        vx = data[:, 3]   # x方向の速度
        vy = data[:, 4]   # y方向の速度
        
        # 運動エネルギーを計算: E = (1/2) * m * (vx² + vy²)
        energy = 0.5 * m * (vx**2 + vy**2)
        # 全時刻のエネルギー値をリストに追加
        energies.extend(energy)
    
    # この温度での全エネルギー値を配列に変換してリストに追加
    energies_by_T.append((T, np.array(energies)))

# 第1図: シミュレーション結果のみのヒストグラム
plt.figure(figsize=(10, 7))
colors = ['blue', 'red', 'green']  # 各温度に対応する色

# 各温度についてヒストグラムを描画
for idx, (T, energies) in enumerate(energies_by_T):
    # ヒストグラムを描画（ビン数30、密度正規化、透明度0.6、黒い枠線）
    plt.hist(energies, bins=30, density=True, alpha=0.6, 
             color=colors[idx], label=f'T={T}', edgecolor='black')

# 軸ラベルとタイトルを設定
plt.xlabel('運動エネルギー E')      # x軸ラベル
plt.ylabel('確率密度')              # y軸ラベル
plt.title('粒子の運動エネルギー分布関数')  # タイトル
plt.legend()     # 凡例を表示
plt.grid(True, alpha=0.3)  # グリッドを表示（透明度0.3）

# レイアウトを調整
plt.tight_layout()
# 図をファイルに保存（解像度150dpi）
plt.savefig(os.path.join('figures', 'energy_distribution.png'), dpi=150)
# メモリを解放するために図を閉じる
plt.close()

# 第2図: シミュレーション結果と理論値の比較
plt.figure(figsize=(10, 7))

# 各温度についてヒストグラムと理論曲線を描画
for idx, (T, energies) in enumerate(energies_by_T):
    # シミュレーション結果のヒストグラムを描画
    plt.hist(energies, bins=30, density=True, alpha=0.6, 
             color=colors[idx], label=f'シミュレーション T={T}', edgecolor='black')
    
    # 理論的なボルツマン分布を計算
    # エネルギー範囲を生成（0から最大値まで1000点）
    E_range = np.linspace(0, energies.max(), 1000)
    # ボルツマン分布: P(E) = (1/kBT) * exp(-E/kBT)
    P_theory = (1.0 / (kB * T)) * np.exp(-E_range / (kB * T))
    
    # 理論曲線を破線で描画
    plt.plot(E_range, P_theory, '--', linewidth=2, 
             color=colors[idx], alpha=0.8, label=f'理論値 T={T}')

# 軸ラベルとタイトルを設定
plt.xlabel('運動エネルギー E')      # x軸ラベル
plt.ylabel('確率密度')              # y軸ラベル
plt.title('粒子の運動エネルギー分布関数（理論値との比較）')  # タイトル
plt.legend()     # 凡例を表示
plt.grid(True, alpha=0.3)  # グリッドを表示（透明度0.3）

# レイアウトを調整
plt.tight_layout()
# 図をファイルに保存（解像度150dpi）
plt.savefig(os.path.join('figures', 'energy_distribution_with_theory.png'), dpi=150)
# メモリを解放するために図を閉じる
plt.close()

# 解析完了のメッセージを出力
print("エネルギー分布関数の解析が完了しました。")
print(f"温度 T = {T_values} について、各{n_runs}回のシミュレーション結果を統合しました。")
