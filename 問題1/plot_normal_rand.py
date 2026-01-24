import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os

plt.rcParams['font.family'] = 'Hiragino Sans'
plt.rcParams['axes.unicode_minus'] = False

os.makedirs('data', exist_ok=True)
os.makedirs('figures', exist_ok=True)

# 50, 100, 1000回の正規乱数を生成
n_samples_list = [50, 100, 1000]
data_list = []

for n_samples in n_samples_list:
    filename = f'normal_rand_{n_samples}.dat' if n_samples != 1000 else 'normal_rand.dat'
    filepath = os.path.join('data', filename)
    
    # Cプログラムを実行してデータを生成
    with open(filepath, 'w') as f:
        subprocess.run(['./normal_rand', str(n_samples)], stdout=f)
    
    data = np.loadtxt(filepath)
    data_list.append((n_samples, data))

# 3つのヒストグラムを表示
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
