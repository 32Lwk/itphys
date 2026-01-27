"""
plot_msd_parameters.py

目的: 平均二乗変位（MSD）のパラメータ依存性を可視化
- 温度T、質量m、摩擦係数γを変化させてMSDを計算
- 各パラメータ値についてシミュレーション結果と理論値を比較

処理の流れ:
1. 各パラメータ（温度、質量、摩擦係数）について複数の値を設定
2. 各パラメータ値についてブラウン運動をシミュレート
3. MSDを計算し、理論値と比較
4. 各試行のMSD、平均MSD、理論MSDをプロット
5. 結果を図として保存
"""

import numpy as np          # 数値計算ライブラリ
import matplotlib.pyplot as plt  # グラフ描画ライブラリ
import os                   # ファイル操作用

# 日本語フォントの設定
plt.rcParams['font.family'] = 'Hiragino Sans'
# マイナス記号の文字化けを防ぐ設定
plt.rcParams['axes.unicode_minus'] = False

def simulate_brownian_motion(T, m, gamma, kB=1.0, dt=0.01, n_steps=1000, seed=None):
    """
    ブラウン運動を数値的にシミュレートする関数
    
    @param T: 温度
    @param m: 粒子の質量
    @param gamma: 摩擦係数
    @param kB: ボルツマン定数（デフォルト: 1.0）
    @param dt: 時間刻み（デフォルト: 0.01）
    @param n_steps: 時間ステップ数（デフォルト: 1000）
    @param seed: 乱数のシード（デフォルト: None）
    @return: (t, x, y) のタプル（時刻、x座標、y座標の配列）
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
    
    # 初期値を設定
    t[0] = 0.0
    x[0] = rx
    y[0] = ry
    
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
    
    return t, x, y

def calculate_msd_from_trajectories(trajectories):
    """
    複数の軌道から平均二乗変位（MSD）を計算する関数
    
    @param trajectories: 軌道のリスト（各要素は(t, x, y)のタプル）
    @return: (t, msd, msd_individual) のタプル
             - t: 時刻の配列
             - msd: 平均二乗変位の配列
             - msd_individual: 各試行のMSDのリスト
    """
    t = trajectories[0][0]  # 最初の軌道から時刻の配列を取得
    n_times = len(t)        # 時刻の数
    n_runs = len(trajectories)  # 軌道の数（実行回数）
    
    # 各試行のMSDを計算
    msd_individual = []
    for _, x, y in trajectories:
        # 各時刻での位置の2乗（x² + y²）を計算
        msd_individual.append(x**2 + y**2)
    
    # 全試行の平均MSDを計算
    msd = np.array([np.mean([msd_individual[j][i] for j in range(n_runs)]) 
                    for i in range(n_times)])
    
    return t, msd, msd_individual

def theoretical_msd(t, T, m, gamma, kB):
    """
    理論的な平均二乗変位を計算する関数
    
    @param t: 時刻の配列
    @param T: 温度
    @param m: 粒子の質量
    @param gamma: 摩擦係数
    @param kB: ボルツマン定数
    @return: (msd_theory, msd_diffusion, D) のタプル
             - msd_theory: 理論的なMSD（短時間と長時間の両方を含む一般式）
             - msd_diffusion: 長時間極限でのMSD（拡散領域: MSD = 4Dt）
             - D: 拡散係数
    """
    tau = m / gamma  # 緩和時間
    # 理論的なMSD（短時間と長時間の両方を含む一般式）
    msd_theory = (4.0 * kB * T / gamma) * (t - tau * (1.0 - np.exp(-t / tau)))
    D = kB * T / gamma  # 拡散係数（アインシュタインの関係式）
    # 長時間極限でのMSD（拡散領域: MSD = 4Dt）
    msd_diffusion = 4.0 * D * t
    return msd_theory, msd_diffusion, D

# 図を保存するディレクトリを作成
os.makedirs('figures', exist_ok=True)

# シミュレーションパラメータ
kB = 1.0      # ボルツマン定数
dt = 0.01     # 時間刻み
n_steps = 1000  # 時間ステップ数
n_runs = 5    # 各パラメータ値についての実行回数

# 解析するパラメータ値のリスト
T_values = [0.5, 1.0, 2.0, 5.0]      # 温度のリスト
m_values = [0.5, 1.0, 2.0]            # 質量のリスト
gamma_values = [0.5, 1.0, 2.0]        # 摩擦係数のリスト

# 3つのサブプロットを持つ図を作成（横に3つ並べる、サイズは18x5インチ）
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# ========== 温度依存性のプロット ==========
ax = axes[0]
# 各温度に異なる色を割り当て（viridisカラーマップを使用）
colors_T = plt.cm.viridis(np.linspace(0, 1, len(T_values)))

# 各温度について
for idx, T in enumerate(T_values):
    m, gamma = 1.0, 1.0  # 質量と摩擦係数は固定
    
    # 複数回のシミュレーションを実行
    trajectories = []
    for run in range(n_runs):
        t, x, y = simulate_brownian_motion(T, m, gamma, kB, dt, n_steps, seed=run)
        trajectories.append((t, x, y))
    
    # MSDを計算
    t, msd, msd_individual = calculate_msd_from_trajectories(trajectories)
    # 理論値を計算
    msd_theory, msd_diffusion, D = theoretical_msd(t, T, m, gamma, kB)
    
    # 各試行のMSDを薄い線で描画（最大3試行まで）
    for i in range(min(3, n_runs)):
        ax.plot(t, msd_individual[i], '-', linewidth=1, alpha=0.3, color=colors_T[idx])
    
    # 平均MSDを太い線で描画
    ax.plot(t, msd, '-', linewidth=2.5, color=colors_T[idx], label=f'T={T}')
    # 理論値を破線で描画
    ax.plot(t, msd_theory, '--', linewidth=1.5, color=colors_T[idx], alpha=0.6)

# 軸ラベルとタイトルを設定
ax.set_xlabel('時間 t')                    # x軸ラベル
ax.set_ylabel('平均二乗変位 <r²(t)>')     # y軸ラベル
ax.set_title('温度依存性 (m=1.0, γ=1.0)')  # タイトル
ax.legend()     # 凡例を表示
ax.grid(True, alpha=0.3)  # グリッドを表示

# ========== 質量依存性のプロット ==========
ax = axes[1]
# 各質量に異なる色を割り当て（plasmaカラーマップを使用）
colors_m = plt.cm.plasma(np.linspace(0, 1, len(m_values)))

# 各質量について
for idx, m in enumerate(m_values):
    T, gamma = 1.0, 1.0  # 温度と摩擦係数は固定
    
    # 複数回のシミュレーションを実行
    trajectories = []
    for run in range(n_runs):
        t, x, y = simulate_brownian_motion(T, m, gamma, kB, dt, n_steps, seed=run)
        trajectories.append((t, x, y))
    
    # MSDを計算
    t, msd, msd_individual = calculate_msd_from_trajectories(trajectories)
    # 理論値を計算
    msd_theory, msd_diffusion, D = theoretical_msd(t, T, m, gamma, kB)
    
    # 各試行のMSDを薄い線で描画（最大3試行まで）
    for i in range(min(3, n_runs)):
        ax.plot(t, msd_individual[i], '-', linewidth=1, alpha=0.3, color=colors_m[idx])
    
    # 平均MSDを太い線で描画
    ax.plot(t, msd, '-', linewidth=2.5, color=colors_m[idx], label=f'm={m}')
    # 理論値を破線で描画
    ax.plot(t, msd_theory, '--', linewidth=1.5, color=colors_m[idx], alpha=0.6)

# 軸ラベルとタイトルを設定
ax.set_xlabel('時間 t')                    # x軸ラベル
ax.set_ylabel('平均二乗変位 <r²(t)>')     # y軸ラベル
ax.set_title('質量依存性 (T=1.0, γ=1.0)')  # タイトル
ax.legend()     # 凡例を表示
ax.grid(True, alpha=0.3)  # グリッドを表示

# ========== 摩擦係数依存性のプロット ==========
ax = axes[2]
# 各摩擦係数に異なる色を割り当て（infernoカラーマップを使用）
colors_gamma = plt.cm.inferno(np.linspace(0, 1, len(gamma_values)))

# 各摩擦係数について
for idx, gamma in enumerate(gamma_values):
    T, m = 1.0, 1.0  # 温度と質量は固定
    
    # 複数回のシミュレーションを実行
    trajectories = []
    for run in range(n_runs):
        t, x, y = simulate_brownian_motion(T, m, gamma, kB, dt, n_steps, seed=run)
        trajectories.append((t, x, y))
    
    # MSDを計算
    t, msd, msd_individual = calculate_msd_from_trajectories(trajectories)
    # 理論値を計算
    msd_theory, msd_diffusion, D = theoretical_msd(t, T, m, gamma, kB)
    
    # 各試行のMSDを薄い線で描画（最大3試行まで）
    for i in range(min(3, n_runs)):
        ax.plot(t, msd_individual[i], '-', linewidth=1, alpha=0.3, color=colors_gamma[idx])
    
    # 平均MSDを太い線で描画
    ax.plot(t, msd, '-', linewidth=2.5, color=colors_gamma[idx], label=f'γ={gamma}')
    # 理論値を破線で描画
    ax.plot(t, msd_theory, '--', linewidth=1.5, color=colors_gamma[idx], alpha=0.6)

# 軸ラベルとタイトルを設定
ax.set_xlabel('時間 t')                    # x軸ラベル
ax.set_ylabel('平均二乗変位 <r²(t)>')     # y軸ラベル
ax.set_title('摩擦係数依存性 (T=1.0, m=1.0)')  # タイトル
ax.legend()     # 凡例を表示
ax.grid(True, alpha=0.3)  # グリッドを表示

# レイアウトを調整
plt.tight_layout()
# 図をファイルに保存（解像度150dpi）
plt.savefig('figures/msd_parameter_dependence.png', dpi=150)
# メモリを解放するために図を閉じる
plt.close()

# 完了メッセージを出力
print("各パラメータについての平均二乗変位の図を生成しました。")
