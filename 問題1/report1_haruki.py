"""
report1_haruki.py

問題1に必要な解析・可視化を一つのスクリプトにまとめたもの（Pythonのみで完結）

内容:
1. 正規分布乱数のヒストグラムと理論曲線の比較
2. ブラウン運動の2次元軌道の可視化
3. 平均二乗変位（MSD）の可視化
4. MSDのパラメータ（温度・質量・摩擦係数）依存性
5. 拡散係数のパラメータ依存性（理論値 D=kBT/γ との比較）
6. 運動エネルギー分布とボルツマン分布の比較

実行: python report1_haruki.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# 日本語フォントの設定
plt.rcParams['font.family'] = 'Hiragino Sans'
plt.rcParams['axes.unicode_minus'] = False

# ========== 共通: ブラウン運動シミュレーション ==========

def simulate_brownian_motion(T=1.0, m=1.0, gamma=1.0, kB=1.0, dt=0.01, n_steps=1000, seed=None):
    """
    ランジュバン方程式に基づく2次元ブラウン運動をシミュレート
    戻り値: (t, x, y, vx, vy) のタプル
    """
    if seed is not None:
        np.random.seed(seed)
    rx, ry = 0.0, 0.0
    vx, vy = 0.0, 0.0
    coeff1 = gamma / m
    coeff2 = np.sqrt(2.0 * gamma * kB * T / m)

    t = np.zeros(n_steps + 1)
    x = np.zeros(n_steps + 1)
    y = np.zeros(n_steps + 1)
    vx_arr = np.zeros(n_steps + 1)
    vy_arr = np.zeros(n_steps + 1)
    t[0], x[0], y[0] = 0.0, rx, ry
    vx_arr[0], vy_arr[0] = vx, vy

    for n in range(n_steps):
        eta_x = np.random.normal(0, 1)
        eta_y = np.random.normal(0, 1)
        vx = vx - coeff1 * vx * dt + coeff2 * np.sqrt(dt) * eta_x
        vy = vy - coeff1 * vy * dt + coeff2 * np.sqrt(dt) * eta_y
        rx += vx * dt
        ry += vy * dt
        t[n+1] = (n+1) * dt
        x[n+1], y[n+1] = rx, ry
        vx_arr[n+1], vy_arr[n+1] = vx, vy
    return t, x, y, vx_arr, vy_arr


def calculate_msd_from_trajectories(trajectories):
    """軌道のリスト [(t,x,y), ...] から平均二乗変位 (t, msd) を計算"""
    t = trajectories[0][0]
    n_times, n_runs = len(t), len(trajectories)
    msd = np.zeros(n_times)
    for t_idx in range(n_times):
        r2_sum = sum(x[t_idx]**2 + y[t_idx]**2 for _, x, y in trajectories)
        msd[t_idx] = r2_sum / n_runs
    return t, msd


def calculate_msd_with_individual(trajectories):
    """軌道からMSDと各試行のMSDを返す: (t, msd, msd_individual)"""
    t = trajectories[0][0]
    n_times = len(t)
    n_runs = len(trajectories)
    msd_individual = [x**2 + y**2 for _, x, y in trajectories]
    msd = np.array([np.mean([msd_individual[j][i] for j in range(n_runs)]) for i in range(n_times)])
    return t, msd, msd_individual


def fit_diffusion_coefficient(t, msd, t_start=None, t_end=None):
    """MSDから拡散係数 D をフィッティング（長時間極限 MSD = 4Dt）"""
    if t_start is None:
        t_start = t[len(t)//2]
    if t_end is None:
        t_end = t[-1]
    mask = (t >= t_start) & (t <= t_end)
    t_fit, msd_fit = t[mask], msd[mask]
    return np.mean(msd_fit / (4.0 * t_fit))


def theoretical_msd(t, T, m, gamma, kB):
    """理論MSDと拡散係数: (msd_theory, msd_diffusion, D)"""
    tau = m / gamma
    msd_theory = (4.0 * kB * T / gamma) * (t - tau * (1.0 - np.exp(-t / tau)))
    D = kB * T / gamma
    msd_diffusion = 4.0 * D * t
    return msd_theory, msd_diffusion, D


# ========== 1. 正規分布乱数のヒストグラム ==========

def run_normal_rand_hist():
    n_samples_list = [50, 100, 1000]
    data_list = []
    for n_samples in n_samples_list:
        data = np.random.normal(0, 1, n_samples)
        data_list.append((n_samples, data))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, (n_samples, data) in enumerate(data_list):
        axes[idx].hist(data, bins=20, density=True, alpha=0.7, edgecolor='black')
        x = np.linspace(data.min(), data.max(), 1000)
        theoretical = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * x**2)
        axes[idx].plot(x, theoretical, 'r-', linewidth=2, label='理論値 N(0,1)')
        axes[idx].set_xlabel('値')
        axes[idx].set_ylabel('確率密度')
        axes[idx].set_title(f'正規分布乱数のヒストグラム (n={n_samples})')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join('figures', 'normal_rand_hist_all.png'), dpi=150)
    plt.close()
    print("1. 正規分布乱数のヒストグラムを保存: figures/normal_rand_hist_all.png")


# ========== 2. 2次元軌道の可視化 ==========

def run_visualize_trajectories(n_runs=5):
    trajectories = []
    for i in range(n_runs):
        t, x, y, _, _ = simulate_brownian_motion(seed=i)
        trajectories.append((t, x, y))

    plt.figure(figsize=(10, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, n_runs))
    all_x = np.concatenate([x for _, x, _ in trajectories])
    all_y = np.concatenate([y for _, _, y in trajectories])
    for i, (t, x, y) in enumerate(trajectories):
        plt.plot(x, y, '-', linewidth=2.0, alpha=0.8, color=colors[i], label=f'実行 {i+1}')
        plt.plot(x[0], y[0], 'o', markersize=10, color=colors[i], markeredgecolor='black', markeredgewidth=1)
        plt.plot(x[-1], y[-1], 's', markersize=10, color=colors[i], markeredgecolor='black', markeredgewidth=1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'ブラウン運動の2次元軌道 ({n_runs}回実行)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    margin = 0.1
    plt.xlim(all_x.min() - margin * (all_x.max() - all_x.min()),
             all_x.max() + margin * (all_x.max() - all_x.min()))
    plt.ylim(all_y.min() - margin * (all_y.max() - all_y.min()),
             all_y.max() + margin * (all_y.max() - all_y.min()))
    plt.tight_layout()
    plt.savefig(os.path.join('figures', 'trajectories_2d.png'), dpi=150)
    plt.close()
    print("2. 2次元軌道を保存: figures/trajectories_2d.png")


# ========== 3. 平均二乗変位の可視化 ==========

def run_visualize_msd(n_runs=5, T=1.0, m=1.0, gamma=1.0, kB=1.0):
    trajectories = []
    for i in range(n_runs):
        t, x, y, _, _ = simulate_brownian_motion(T, m, gamma, kB, seed=i)
        trajectories.append((t, x, y))
    t, msd, msd_individual = calculate_msd_with_individual(trajectories)

    plt.figure(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, n_runs))
    for i in range(n_runs):
        plt.plot(t, msd_individual[i], '-', linewidth=1.5, alpha=0.6, color=colors[i], label=f'試行 {i+1}')
    plt.plot(t, msd, 'k-', linewidth=3, label='平均 <r²(t)>', alpha=0.9)
    plt.xlabel('時間 t')
    plt.ylabel('平均二乗変位 <r²(t)>')
    plt.title(f'平均二乗変位 (n={n_runs}回実行, T={T}, m={m}, γ={gamma})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join('figures', 'msd_plot.png'), dpi=150)
    plt.close()
    print("3. MSDプロットを保存: figures/msd_plot.png")


# ========== 4. MSDのパラメータ依存性 ==========

def run_msd_parameter_dependence(kB=1.0, dt=0.01, n_steps=1000, n_runs=5):
    T_values = [0.5, 1.0, 2.0, 5.0]
    m_values = [0.5, 1.0, 2.0]
    gamma_values = [0.5, 1.0, 2.0]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors_T = plt.cm.viridis(np.linspace(0, 1, len(T_values)))
    colors_m = plt.cm.plasma(np.linspace(0, 1, len(m_values)))
    colors_gamma = plt.cm.inferno(np.linspace(0, 1, len(gamma_values)))

    for idx, T in enumerate(T_values):
        m, gamma = 1.0, 1.0
        trajectories = [simulate_brownian_motion(T, m, gamma, kB, dt, n_steps, seed=r) for r in range(n_runs)]
        trajectories = [(tr[0], tr[1], tr[2]) for tr in trajectories]
        t, msd, msd_individual = calculate_msd_with_individual(trajectories)
        msd_theory, _, _ = theoretical_msd(t, T, m, gamma, kB)
        for i in range(min(3, n_runs)):
            axes[0].plot(t, msd_individual[i], '-', linewidth=1, alpha=0.3, color=colors_T[idx])
        axes[0].plot(t, msd, '-', linewidth=2.5, color=colors_T[idx], label=f'T={T}')
        axes[0].plot(t, msd_theory, '--', linewidth=1.5, color=colors_T[idx], alpha=0.6)
    axes[0].set_xlabel('時間 t')
    axes[0].set_ylabel('平均二乗変位 <r²(t)>')
    axes[0].set_title('温度依存性 (m=1.0, γ=1.0)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for idx, m in enumerate(m_values):
        T, gamma = 1.0, 1.0
        trajectories = [simulate_brownian_motion(T, m, gamma, kB, dt, n_steps, seed=r) for r in range(n_runs)]
        trajectories = [(tr[0], tr[1], tr[2]) for tr in trajectories]
        t, msd, msd_individual = calculate_msd_with_individual(trajectories)
        msd_theory, _, _ = theoretical_msd(t, T, m, gamma, kB)
        for i in range(min(3, n_runs)):
            axes[1].plot(t, msd_individual[i], '-', linewidth=1, alpha=0.3, color=colors_m[idx])
        axes[1].plot(t, msd, '-', linewidth=2.5, color=colors_m[idx], label=f'm={m}')
        axes[1].plot(t, msd_theory, '--', linewidth=1.5, color=colors_m[idx], alpha=0.6)
    axes[1].set_xlabel('時間 t')
    axes[1].set_ylabel('平均二乗変位 <r²(t)>')
    axes[1].set_title('質量依存性 (T=1.0, γ=1.0)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    for idx, gamma in enumerate(gamma_values):
        T, m = 1.0, 1.0
        trajectories = [simulate_brownian_motion(T, m, gamma, kB, dt, n_steps, seed=r) for r in range(n_runs)]
        trajectories = [(tr[0], tr[1], tr[2]) for tr in trajectories]
        t, msd, msd_individual = calculate_msd_with_individual(trajectories)
        msd_theory, _, _ = theoretical_msd(t, T, m, gamma, kB)
        for i in range(min(3, n_runs)):
            axes[2].plot(t, msd_individual[i], '-', linewidth=1, alpha=0.3, color=colors_gamma[idx])
        axes[2].plot(t, msd, '-', linewidth=2.5, color=colors_gamma[idx], label=f'γ={gamma}')
        axes[2].plot(t, msd_theory, '--', linewidth=1.5, color=colors_gamma[idx], alpha=0.6)
    axes[2].set_xlabel('時間 t')
    axes[2].set_ylabel('平均二乗変位 <r²(t)>')
    axes[2].set_title('摩擦係数依存性 (T=1.0, m=1.0)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join('figures', 'msd_parameter_dependence.png'), dpi=150)
    plt.close()
    print("4. MSDパラメータ依存性を保存: figures/msd_parameter_dependence.png")


# ========== 5. 拡散係数のパラメータ依存性 ==========

def run_diffusion_parameter_dependence(kB=1.0, dt=0.01, n_steps=1000, n_runs=5):
    T_values = [0.5, 1.0, 2.0, 5.0]
    m_values = [0.5, 1.0, 2.0]
    gamma_values = [0.5, 1.0, 2.0]

    T_results, m_results, gamma_results = [], [], []

    for T in T_values:
        m, gamma = 1.0, 1.0
        D_theory = kB * T / gamma
        trajectories = [simulate_brownian_motion(T, m, gamma, kB, dt, n_steps, seed=r) for r in range(n_runs)]
        trajectories = [(tr[0], tr[1], tr[2]) for tr in trajectories]
        t, msd = calculate_msd_from_trajectories(trajectories)
        D_fit = fit_diffusion_coefficient(t, msd)
        T_results.append((T, D_theory, D_fit))

    for m in m_values:
        T, gamma = 1.0, 1.0
        D_theory = kB * T / gamma
        trajectories = [simulate_brownian_motion(T, m, gamma, kB, dt, n_steps, seed=r) for r in range(n_runs)]
        trajectories = [(tr[0], tr[1], tr[2]) for tr in trajectories]
        t, msd = calculate_msd_from_trajectories(trajectories)
        D_fit = fit_diffusion_coefficient(t, msd)
        m_results.append((m, D_theory, D_fit))

    for gamma in gamma_values:
        T, m = 1.0, 1.0
        D_theory = kB * T / gamma
        trajectories = [simulate_brownian_motion(T, m, gamma, kB, dt, n_steps, seed=r) for r in range(n_runs)]
        trajectories = [(tr[0], tr[1], tr[2]) for tr in trajectories]
        t, msd = calculate_msd_from_trajectories(trajectories)
        D_fit = fit_diffusion_coefficient(t, msd)
        gamma_results.append((gamma, D_theory, D_fit))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for vals, ax, xlabel, title in [
        (T_results, axes[0], '温度 T', '温度依存性 (m=1.0, γ=1.0)'),
        (m_results, axes[1], '質量 m', '質量依存性 (T=1.0, γ=1.0)'),
        (gamma_results, axes[2], '摩擦係数 γ', '摩擦係数依存性 (T=1.0, m=1.0)'),
    ]:
        x_vals = [v[0] for v in vals]
        D_th = [v[1] for v in vals]
        D_fit = [v[2] for v in vals]
        ax.plot(x_vals, D_th, 'ro-', markersize=10, label='理論値 D=kB*T/γ', linewidth=2)
        ax.plot(x_vals, D_fit, 'bs--', markersize=8, label='フィッティング値 D', linewidth=2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('拡散係数 D')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join('figures', 'diffusion_parameter_dependence.png'), dpi=150)
    plt.close()
    print("5. 拡散係数パラメータ依存性を保存: figures/diffusion_parameter_dependence.png")


# ========== 6. 運動エネルギー分布とボルツマン分布 ==========

def run_energy_distribution(T_values=None, m=1.0, gamma=1.0, kB=1.0, dt=0.01, n_steps=1000, n_runs=10):
    if T_values is None:
        T_values = [0.5, 1.0, 2.0]
    energies_by_T = []
    for T in T_values:
        energies = []
        for run in range(n_runs):
            t, x, y, vx, vy = simulate_brownian_motion(T, m, gamma, kB, dt, n_steps, seed=run)
            energy = 0.5 * m * (vx**2 + vy**2)
            energies.extend(energy)
        energies_by_T.append((T, np.array(energies)))

    colors = ['blue', 'red', 'green']
    plt.figure(figsize=(10, 7))
    for idx, (T, energies) in enumerate(energies_by_T):
        plt.hist(energies, bins=30, density=True, alpha=0.6, color=colors[idx], label=f'T={T}', edgecolor='black')
    plt.xlabel('運動エネルギー E')
    plt.ylabel('確率密度')
    plt.title('粒子の運動エネルギー分布関数')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join('figures', 'energy_distribution.png'), dpi=150)
    plt.close()

    plt.figure(figsize=(10, 7))
    for idx, (T, energies) in enumerate(energies_by_T):
        plt.hist(energies, bins=30, density=True, alpha=0.6, color=colors[idx],
                 label=f'シミュレーション T={T}', edgecolor='black')
        E_range = np.linspace(0, energies.max(), 1000)
        P_theory = (1.0 / (kB * T)) * np.exp(-E_range / (kB * T))
        plt.plot(E_range, P_theory, '--', linewidth=2, color=colors[idx], alpha=0.8, label=f'理論値 T={T}')
    plt.xlabel('運動エネルギー E')
    plt.ylabel('確率密度')
    plt.title('粒子の運動エネルギー分布関数（理論値との比較）')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join('figures', 'energy_distribution_with_theory.png'), dpi=150)
    plt.close()
    print("6. エネルギー分布を保存: figures/energy_distribution.png, figures/energy_distribution_with_theory.png")


# ========== メイン ==========

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    os.makedirs('data', exist_ok=True)
    os.makedirs('figures', exist_ok=True)

    n_runs_traj = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    T_msd = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0

    run_normal_rand_hist()
    run_visualize_trajectories(n_runs=n_runs_traj)
    run_visualize_msd(n_runs=n_runs_traj, T=T_msd)
    run_msd_parameter_dependence()
    run_diffusion_parameter_dependence()
    run_energy_distribution()

    print("問題1の解析がすべて完了しました。")


if __name__ == '__main__':
    main()
