import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os
import sys

plt.rcParams['font.family'] = 'Hiragino Sans'
plt.rcParams['axes.unicode_minus'] = False

n_runs = int(sys.argv[1]) if len(sys.argv) > 1 else 5
os.makedirs('data', exist_ok=True)
os.makedirs('figures', exist_ok=True)

trajectories = []
for i in range(n_runs):
    output_file = os.path.join('data', f'trajectory_{i+1}.dat')
    with open(output_file, 'w') as f:
        subprocess.run(['./brownian_motion'], stdout=f)
    data = np.loadtxt(output_file, comments='#')
    trajectories.append((data[:, 0], data[:, 1], data[:, 2]))

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
