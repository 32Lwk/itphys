"""
analyze_diffusion.py

目的: 拡散係数のパラメータ依存性を解析
- 温度T、質量m、摩擦係数γを変化させて拡散係数Dを計算
- 理論値（D = kBT/γ）とシミュレーション結果を比較

処理の流れ:
1. 各パラメータ（温度、質量、摩擦係数）について複数の値を設定
2. 各パラメータ値についてブラウン運動をシミュレート
3. MSDから拡散係数をフィッティング
4. 理論値とフィッティング値を比較してプロット
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
    @return: (t, msd) のタプル（時刻、平均二乗変位の配列）
    """
    t = trajectories[0][0]  # 最初の軌道から時刻の配列を取得
    n_times = len(t)        # 時刻の数
    n_runs = len(trajectories)  # 軌道の数（実行回数）
    
    # MSDの配列を初期化
    msd = np.zeros(n_times)
    
    # 各時刻について
    for t_idx in range(n_times):
        r2_sum = 0.0  # 全軌道のr²の合計
        # 全軌道について
        for _, x, y in trajectories:
            # この時刻での位置の2乗（x² + y²）を合計に加える
            r2_sum += x[t_idx]**2 + y[t_idx]**2
        # 平均を計算
        msd[t_idx] = r2_sum / n_runs
    
    return t, msd

def fit_diffusion_coefficient(t, msd, t_start=None, t_end=None):
    """
    MSDから拡散係数をフィッティングする関数
    長時間極限では MSD = 4Dt となるため、D = MSD/(4t) から拡散係数を推定
    
    @param t: 時刻の配列
    @param msd: 平均二乗変位の配列
    @param t_start: フィッティング開始時刻（デフォルト: 中間時刻）
    @param t_end: フィッティング終了時刻（デフォルト: 最終時刻）
    @return: フィッティングされた拡散係数D
    """
    # フィッティング範囲を決定
    if t_start is None:
        t_start = t[len(t)//2]  # デフォルト: 中間時刻
    if t_end is None:
        t_end = t[-1]  # デフォルト: 最終時刻
    
    # フィッティング範囲のマスクを作成
    mask = (t >= t_start) & (t <= t_end)
    t_fit = t[mask]      # フィッティング範囲の時刻
    msd_fit = msd[mask]  # フィッティング範囲のMSD
    
    # D = MSD/(4t) の平均値を拡散係数として使用
    D_fit = np.mean(msd_fit / (4.0 * t_fit))
    
    return D_fit

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

# ========== 温度依存性の解析 ==========
print("=" * 60)
print("温度依存性 (m=1.0, γ=1.0)")
print("=" * 60)
T_results = []  # 結果を格納するリスト

# 各温度について
for T in T_values:
    m, gamma = 1.0, 1.0  # 質量と摩擦係数は固定
    
    # 理論的な拡散係数（アインシュタインの関係式: D = kBT/γ）
    D_theory = kB * T / gamma
    
    # 複数回のシミュレーションを実行
    trajectories = []
    for run in range(n_runs):
        t, x, y = simulate_brownian_motion(T, m, gamma, kB, dt, n_steps, seed=run)
        trajectories.append((t, x, y))
    
    # MSDを計算
    t, msd = calculate_msd_from_trajectories(trajectories)
    
    # 拡散係数をフィッティング
    D_fit = fit_diffusion_coefficient(t, msd)
    
    # 結果を保存
    T_results.append((T, D_theory, D_fit))
    
    # 結果を出力
    error = abs(D_theory - D_fit) / D_theory * 100  # 相対誤差（%）
    print(f"T={T:.2f}: D_theory={D_theory:.6f}, D_fit={D_fit:.6f}, error={error:.2f}%")

# ========== 質量依存性の解析 ==========
print("\n" + "=" * 60)
print("質量依存性 (T=1.0, γ=1.0)")
print("=" * 60)
m_results = []  # 結果を格納するリスト

# 各質量について
for m in m_values:
    T, gamma = 1.0, 1.0  # 温度と摩擦係数は固定
    
    # 理論的な拡散係数（質量には依存しない: D = kBT/γ）
    D_theory = kB * T / gamma
    
    # 複数回のシミュレーションを実行
    trajectories = []
    for run in range(n_runs):
        t, x, y = simulate_brownian_motion(T, m, gamma, kB, dt, n_steps, seed=run)
        trajectories.append((t, x, y))
    
    # MSDを計算
    t, msd = calculate_msd_from_trajectories(trajectories)
    
    # 拡散係数をフィッティング
    D_fit = fit_diffusion_coefficient(t, msd)
    
    # 結果を保存
    m_results.append((m, D_theory, D_fit))
    
    # 結果を出力
    error = abs(D_theory - D_fit) / D_theory * 100  # 相対誤差（%）
    print(f"m={m:.2f}: D_theory={D_theory:.6f}, D_fit={D_fit:.6f}, error={error:.2f}%")

# ========== 摩擦係数依存性の解析 ==========
print("\n" + "=" * 60)
print("摩擦係数依存性 (T=1.0, m=1.0)")
print("=" * 60)
gamma_results = []  # 結果を格納するリスト

# 各摩擦係数について
for gamma in gamma_values:
    T, m = 1.0, 1.0  # 温度と質量は固定
    
    # 理論的な拡散係数（摩擦係数に反比例: D = kBT/γ）
    D_theory = kB * T / gamma
    
    # 複数回のシミュレーションを実行
    trajectories = []
    for run in range(n_runs):
        t, x, y = simulate_brownian_motion(T, m, gamma, kB, dt, n_steps, seed=run)
        trajectories.append((t, x, y))
    
    # MSDを計算
    t, msd = calculate_msd_from_trajectories(trajectories)
    
    # 拡散係数をフィッティング
    D_fit = fit_diffusion_coefficient(t, msd)
    
    # 結果を保存
    gamma_results.append((gamma, D_theory, D_fit))
    
    # 結果を出力
    error = abs(D_theory - D_fit) / D_theory * 100  # 相対誤差（%）
    print(f"γ={gamma:.2f}: D_theory={D_theory:.6f}, D_fit={D_fit:.6f}, error={error:.2f}%")

# ========== 結果の可視化 ==========
# 3つのサブプロットを持つ図を作成（横に3つ並べる、サイズは18x5インチ）
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 温度依存性のプロット
T_vals, D_th, D_fit = zip(*T_results)  # 結果を展開
axes[0].plot(T_vals, D_th, 'ro-', markersize=10, label='理論値 D=kB*T/γ', linewidth=2)
axes[0].plot(T_vals, D_fit, 'bs--', markersize=8, label='フィッティング値 D', linewidth=2)
axes[0].set_xlabel('温度 T')      # x軸ラベル
axes[0].set_ylabel('拡散係数 D')  # y軸ラベル
axes[0].set_title('温度依存性 (m=1.0, γ=1.0)')  # タイトル
axes[0].legend()     # 凡例を表示
axes[0].grid(True, alpha=0.3)  # グリッドを表示

# 質量依存性のプロット
m_vals, D_th, D_fit = zip(*m_results)  # 結果を展開
axes[1].plot(m_vals, D_th, 'ro-', markersize=10, label='理論値 D=kB*T/γ', linewidth=2)
axes[1].plot(m_vals, D_fit, 'bs--', markersize=8, label='フィッティング値 D', linewidth=2)
axes[1].set_xlabel('質量 m')      # x軸ラベル
axes[1].set_ylabel('拡散係数 D')  # y軸ラベル
axes[1].set_title('質量依存性 (T=1.0, γ=1.0)')  # タイトル
axes[1].legend()     # 凡例を表示
axes[1].grid(True, alpha=0.3)  # グリッドを表示

# 摩擦係数依存性のプロット
gamma_vals, D_th, D_fit = zip(*gamma_results)  # 結果を展開
axes[2].plot(gamma_vals, D_th, 'ro-', markersize=10, label='理論値 D=kB*T/γ', linewidth=2)
axes[2].plot(gamma_vals, D_fit, 'bs--', markersize=8, label='フィッティング値 D', linewidth=2)
axes[2].set_xlabel('摩擦係数 γ')  # x軸ラベル
axes[2].set_ylabel('拡散係数 D')  # y軸ラベル
axes[2].set_title('摩擦係数依存性 (T=1.0, m=1.0)')  # タイトル
axes[2].legend()     # 凡例を表示
axes[2].grid(True, alpha=0.3)  # グリッドを表示

# レイアウトを調整
plt.tight_layout()
# 図をファイルに保存（解像度150dpi）
plt.savefig('figures/diffusion_parameter_dependence.png', dpi=150)
# メモリを解放するために図を閉じる
plt.close()
