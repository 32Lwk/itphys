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
    msd_individual = []
    for _, x, y in trajectories:
        msd_individual.append(x**2 + y**2)
    msd = np.array([np.mean([msd_individual[j][i] for j in range(n_runs)]) for i in range(n_times)])
    return t, msd, msd_individual

def theoretical_msd(t, T, m, gamma, kB):
    tau = m / gamma
    msd_theory = (4.0 * kB * T / gamma) * (t - tau * (1.0 - np.exp(-t / tau)))
    D = kB * T / gamma
    msd_diffusion = 4.0 * D * t
    return msd_theory, msd_diffusion, D

os.makedirs('figures', exist_ok=True)

kB, dt, n_steps, n_runs = 1.0, 0.01, 1000, 5
T_values = [0.5, 1.0, 2.0, 5.0]
m_values = [0.5, 1.0, 2.0]
gamma_values = [0.5, 1.0, 2.0]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax = axes[0]
colors_T = plt.cm.viridis(np.linspace(0, 1, len(T_values)))
for idx, T in enumerate(T_values):
    m, gamma = 1.0, 1.0
    trajectories = []
    for run in range(n_runs):
        t, x, y = simulate_brownian_motion(T, m, gamma, kB, dt, n_steps, seed=run)
        trajectories.append((t, x, y))
    t, msd, msd_individual = calculate_msd_from_trajectories(trajectories)
    msd_theory, msd_diffusion, D = theoretical_msd(t, T, m, gamma, kB)
    
    for i in range(min(3, n_runs)):
        ax.plot(t, msd_individual[i], '-', linewidth=1, alpha=0.3, color=colors_T[idx])
    ax.plot(t, msd, '-', linewidth=2.5, color=colors_T[idx], label=f'T={T}')
    ax.plot(t, msd_theory, '--', linewidth=1.5, color=colors_T[idx], alpha=0.6)

ax.set_xlabel('時間 t')
ax.set_ylabel('平均二乗変位 <r²(t)>')
ax.set_title('温度依存性 (m=1.0, γ=1.0)')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
colors_m = plt.cm.plasma(np.linspace(0, 1, len(m_values)))
for idx, m in enumerate(m_values):
    T, gamma = 1.0, 1.0
    trajectories = []
    for run in range(n_runs):
        t, x, y = simulate_brownian_motion(T, m, gamma, kB, dt, n_steps, seed=run)
        trajectories.append((t, x, y))
    t, msd, msd_individual = calculate_msd_from_trajectories(trajectories)
    msd_theory, msd_diffusion, D = theoretical_msd(t, T, m, gamma, kB)
    
    for i in range(min(3, n_runs)):
        ax.plot(t, msd_individual[i], '-', linewidth=1, alpha=0.3, color=colors_m[idx])
    ax.plot(t, msd, '-', linewidth=2.5, color=colors_m[idx], label=f'm={m}')
    ax.plot(t, msd_theory, '--', linewidth=1.5, color=colors_m[idx], alpha=0.6)

ax.set_xlabel('時間 t')
ax.set_ylabel('平均二乗変位 <r²(t)>')
ax.set_title('質量依存性 (T=1.0, γ=1.0)')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[2]
colors_gamma = plt.cm.inferno(np.linspace(0, 1, len(gamma_values)))
for idx, gamma in enumerate(gamma_values):
    T, m = 1.0, 1.0
    trajectories = []
    for run in range(n_runs):
        t, x, y = simulate_brownian_motion(T, m, gamma, kB, dt, n_steps, seed=run)
        trajectories.append((t, x, y))
    t, msd, msd_individual = calculate_msd_from_trajectories(trajectories)
    msd_theory, msd_diffusion, D = theoretical_msd(t, T, m, gamma, kB)
    
    for i in range(min(3, n_runs)):
        ax.plot(t, msd_individual[i], '-', linewidth=1, alpha=0.3, color=colors_gamma[idx])
    ax.plot(t, msd, '-', linewidth=2.5, color=colors_gamma[idx], label=f'γ={gamma}')
    ax.plot(t, msd_theory, '--', linewidth=1.5, color=colors_gamma[idx], alpha=0.6)

ax.set_xlabel('時間 t')
ax.set_ylabel('平均二乗変位 <r²(t)>')
ax.set_title('摩擦係数依存性 (T=1.0, m=1.0)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/msd_parameter_dependence.png', dpi=150)
plt.close()

print("各パラメータについての平均二乗変位の図を生成しました。")
