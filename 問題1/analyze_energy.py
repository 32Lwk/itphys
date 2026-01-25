import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os
import sys

plt.rcParams['font.family'] = 'Hiragino Sans'
plt.rcParams['axes.unicode_minus'] = False

T_values = [0.5, 1.0, 2.0]
m, gamma, kB = 1.0, 1.0, 1.0
dt, n_steps = 0.01, 1000

os.makedirs('data', exist_ok=True)
os.makedirs('figures', exist_ok=True)

energies_by_T = []

for T in T_values:
    energies = []
    n_runs = 10
    for run in range(n_runs):
        output_file = os.path.join('data', f'energy_T{T}_run{run+1}.dat')
        with open(output_file, 'w') as f:
            subprocess.run(['./brownian_motion', str(T), str(m), str(gamma), str(dt), str(n_steps)], stdout=f)
        data = np.loadtxt(output_file, comments='#')
        t = data[:, 0]
        vx = data[:, 3]
        vy = data[:, 4]
        energy = 0.5 * m * (vx**2 + vy**2)
        energies.extend(energy)
    
    energies_by_T.append((T, np.array(energies)))

plt.figure(figsize=(10, 7))
colors = ['blue', 'red', 'green']
for idx, (T, energies) in enumerate(energies_by_T):
    plt.hist(energies, bins=30, density=True, alpha=0.6, 
             color=colors[idx], label=f'T={T}', edgecolor='black')

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
    plt.hist(energies, bins=30, density=True, alpha=0.6, 
             color=colors[idx], label=f'シミュレーション T={T}', edgecolor='black')
    
    E_range = np.linspace(0, energies.max(), 1000)
    P_theory = (1.0 / (kB * T)) * np.exp(-E_range / (kB * T))
    plt.plot(E_range, P_theory, '--', linewidth=2, 
             color=colors[idx], alpha=0.8, label=f'理論値 T={T}')

plt.xlabel('運動エネルギー E')
plt.ylabel('確率密度')
plt.title('粒子の運動エネルギー分布関数（理論値との比較）')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join('figures', 'energy_distribution_with_theory.png'), dpi=150)
plt.close()

print("エネルギー分布関数の解析が完了しました。")
print(f"温度 T = {T_values} について、各{10}回のシミュレーション結果を統合しました。")
