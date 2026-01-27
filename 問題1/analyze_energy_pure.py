"""
analyze_energy_pure.py

目的: ブラウン運動する粒子の運動エネルギー分布関数を解析（Pythonのみ版）
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
import os                   # ファイル操作用

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
        
        # ブラウン運動をシミュレート
        t, x, y, vx, vy = simulate_brownian_motion(T, m, gamma, kB, dt, n_steps, seed=run)
        
        # データをdatファイルに保存（Cプログラムと同じ形式: t x y vx vy）
        data = np.column_stack([t, x, y, vx, vy])
        np.savetxt(output_file, data, fmt='%.15e', header='t x y vx vy', comments='#')
        
        # datファイルからデータを読み込む（#で始まるコメント行は無視）
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
plt.savefig(os.path.join('figures', 'energy_distribution_pure.png'), dpi=150)
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
plt.savefig(os.path.join('figures', 'energy_distribution_with_theory_pure.png'), dpi=150)
# メモリを解放するために図を閉じる
plt.close()

# 解析完了のメッセージを出力
print("エネルギー分布関数の解析が完了しました。")
print(f"温度 T = {T_values} について、各{n_runs}回のシミュレーション結果を統合しました。")
