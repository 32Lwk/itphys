import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.family'] = 'Hiragino Sans'
plt.rcParams['axes.unicode_minus'] = False

def simulate_brownian_motion(T, m, gamma, kB=1.0, dt=0.01, n_steps=1000, seed=None):
    if seed is not None:
        np.random.seed(seed)
    rx, ry = 0.0, 0.0
    vx, vy = 0.0, 0.0
    coeff1 = gamma / m
    coeff2 = np.sqrt(2.0 * gamma * kB * T / m)
    t = np.zeros(n_steps + 1)
    x = np.zeros(n_steps + 1)
    y = np.zeros(n_steps + 1)
    t[0] = 0.0
    x[0] = rx
    y[0] = ry
    for n in range(n_steps):
        eta_x = np.random.normal(0, 1)
        eta_y = np.random.normal(0, 1)
        vx = vx - coeff1 * vx * dt + coeff2 * np.sqrt(dt) * eta_x
        vy = vy - coeff1 * vy * dt + coeff2 * np.sqrt(dt) * eta_y
        rx += vx * dt
        ry += vy * dt
        t[n+1] = (n+1) * dt
        x[n+1] = rx
        y[n+1] = ry
    return t, x, y

def calculate_msd_from_trajectories(trajectories):
    t = trajectories[0][0]
    n_times = len(t)
    n_runs = len(trajectories)
    msd = np.zeros(n_times)
    for t_idx in range(n_times):
        r2_sum = 0.0
        for _, x, y in trajectories:
            r2_sum += x[t_idx]**2 + y[t_idx]**2
        msd[t_idx] = r2_sum / n_runs
    return t, msd

def fit_diffusion_coefficient(t, msd, t_start=None, t_end=None):
    if t_start is None:
        t_start = t[len(t)//2]
    if t_end is None:
        t_end = t[-1]
    mask = (t >= t_start) & (t <= t_end)
    t_fit = t[mask]
    msd_fit = msd[mask]
    D_fit = np.mean(msd_fit / (4.0 * t_fit))
    return D_fit

os.makedirs('figures', exist_ok=True)
    
kB, dt, n_steps, n_runs = 1.0, 0.01, 1000, 5
T_values = [0.5, 1.0, 2.0, 5.0]
m_values = [0.5, 1.0, 2.0]
gamma_values = [0.5, 1.0, 2.0]

print("=" * 60)
print("温度依存性 (m=1.0, γ=1.0)")
print("=" * 60)
T_results = []
for T in T_values:
    m, gamma = 1.0, 1.0
    D_theory = kB * T / gamma
    trajectories = []
    for run in range(n_runs):
        t, x, y = simulate_brownian_motion(T, m, gamma, kB, dt, n_steps, seed=run)
        trajectories.append((t, x, y))
    t, msd = calculate_msd_from_trajectories(trajectories)
    D_fit = fit_diffusion_coefficient(t, msd)
    T_results.append((T, D_theory, D_fit))
    print(f"T={T:.2f}: D_theory={D_theory:.6f}, D_fit={D_fit:.6f}, error={abs(D_theory-D_fit)/D_theory*100:.2f}%")

print("\n" + "=" * 60)
print("質量依存性 (T=1.0, γ=1.0)")
print("=" * 60)
m_results = []
for m in m_values:
    T, gamma = 1.0, 1.0
    D_theory = kB * T / gamma
    trajectories = []
    for run in range(n_runs):
        t, x, y = simulate_brownian_motion(T, m, gamma, kB, dt, n_steps, seed=run)
        trajectories.append((t, x, y))
    t, msd = calculate_msd_from_trajectories(trajectories)
    D_fit = fit_diffusion_coefficient(t, msd)
    m_results.append((m, D_theory, D_fit))
    print(f"m={m:.2f}: D_theory={D_theory:.6f}, D_fit={D_fit:.6f}, error={abs(D_theory-D_fit)/D_theory*100:.2f}%")

print("\n" + "=" * 60)
print("摩擦係数依存性 (T=1.0, m=1.0)")
print("=" * 60)
gamma_results = []
for gamma in gamma_values:
    T, m = 1.0, 1.0
    D_theory = kB * T / gamma
    trajectories = []
    for run in range(n_runs):
        t, x, y = simulate_brownian_motion(T, m, gamma, kB, dt, n_steps, seed=run)
        trajectories.append((t, x, y))
    t, msd = calculate_msd_from_trajectories(trajectories)
    D_fit = fit_diffusion_coefficient(t, msd)
    gamma_results.append((gamma, D_theory, D_fit))
    print(f"γ={gamma:.2f}: D_theory={D_theory:.6f}, D_fit={D_fit:.6f}, error={abs(D_theory-D_fit)/D_theory*100:.2f}%")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
T_vals, D_th, D_fit = zip(*T_results)
axes[0].plot(T_vals, D_th, 'ro-', markersize=10, label='理論値 D=kB*T/γ', linewidth=2)
axes[0].plot(T_vals, D_fit, 'bs--', markersize=8, label='フィッティング値 D', linewidth=2)
axes[0].set_xlabel('温度 T')
axes[0].set_ylabel('拡散係数 D')
axes[0].set_title('温度依存性 (m=1.0, γ=1.0)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

m_vals, D_th, D_fit = zip(*m_results)
axes[1].plot(m_vals, D_th, 'ro-', markersize=10, label='理論値 D=kB*T/γ', linewidth=2)
axes[1].plot(m_vals, D_fit, 'bs--', markersize=8, label='フィッティング値 D', linewidth=2)
axes[1].set_xlabel('質量 m')
axes[1].set_ylabel('拡散係数 D')
axes[1].set_title('質量依存性 (T=1.0, γ=1.0)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

gamma_vals, D_th, D_fit = zip(*gamma_results)
axes[2].plot(gamma_vals, D_th, 'ro-', markersize=10, label='理論値 D=kB*T/γ', linewidth=2)
axes[2].plot(gamma_vals, D_fit, 'bs--', markersize=8, label='フィッティング値 D', linewidth=2)
axes[2].set_xlabel('摩擦係数 γ')
axes[2].set_ylabel('拡散係数 D')
axes[2].set_title('摩擦係数依存性 (T=1.0, m=1.0)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/diffusion_parameter_dependence.png', dpi=150)
plt.close()
