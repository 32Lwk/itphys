import numpy as np
import matplotlib.pyplot as plt
import sys
import os

plt.rcParams['font.family'] = 'Hiragino Sans'
plt.rcParams['axes.unicode_minus'] = False

n_samples = int(sys.argv[1])
os.makedirs('data', exist_ok=True)
os.makedirs('figures', exist_ok=True)

filename = f'normal_rand_{n_samples}.dat' if n_samples != 1000 else 'normal_rand.dat'
data = np.loadtxt(os.path.join('data', filename))

plt.figure(figsize=(8, 6))
plt.hist(data, bins=20, density=True, alpha=0.7, edgecolor='black')
x = np.linspace(data.min(), data.max(), 1000)
theoretical = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * x**2)
plt.plot(x, theoretical, 'r-', linewidth=2, label='理論値 N(0,1)')
plt.xlabel('値')
plt.ylabel('確率密度')
plt.title(f'正規分布乱数のヒストグラム (n={n_samples})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join('figures', f'normal_rand_hist_{n_samples}.png'), dpi=150)
plt.close()
