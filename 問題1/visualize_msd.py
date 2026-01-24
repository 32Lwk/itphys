import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os
import sys

plt.rcParams['font.family'] = 'Hiragino Sans'
plt.rcParams['axes.unicode_minus'] = False

n_runs = int(sys.argv[1]) if len(sys.argv) > 1 else 5
T = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
m, gamma, kB = 1.0, 1.0, 1.0

os.makedirs('data', exist_ok=True)
os.makedirs('figures', exist_ok=True)

trajectories = []
for i in range(n_runs):
    output_file = os.path.join('data', f'trajectory_{i+1}.dat')
    with open(output_file, 'w') as f:
        subprocess.run(['./brownian_motion', str(T), str(m), str(gamma)], stdout=f)
    data = np.loadtxt(output_file, comments='#')
    trajectories.append((data[:, 0], data[:, 1], data[:, 2]))

t = trajectories[0][0]
# 各試行のMSDを計算
msd_individual = []
for _, x, y in trajectories:
    msd_individual.append(x**2 + y**2)

# 平均MSDを計算
msd = np.array([np.mean([msd_individual[j][i] for j in range(n_runs)]) for i in range(len(t))])

D = kB * T / gamma
tau = m / gamma
msd_theory = (4.0 * kB * T / gamma) * (t - tau * (1.0 - np.exp(-t / tau)))
msd_diffusion = 4.0 * D * t

plt.figure(figsize=(10, 8))
# 5回の試行を個別に重ねて表示
colors = plt.cm.tab10(np.linspace(0, 1, n_runs))
for i in range(n_runs):
    plt.plot(t, msd_individual[i], '-', linewidth=1.5, alpha=0.6, 
             color=colors[i], label=f'試行 {i+1}')

# 平均を太い線で表示
plt.plot(t, msd, 'k-', linewidth=3, label='平均 <r²(t)>', alpha=0.9)

plt.xlabel('時間 t')
plt.ylabel('平均二乗変位 <r²(t)>')
plt.title(f'平均二乗変位 (n={n_runs}回実行, T={T}, m={m}, γ={gamma})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join('figures', 'msd_plot.png'), dpi=150)
plt.close()
