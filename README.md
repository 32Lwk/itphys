# 数値シミュレーションによるブラウン運動の解析

## 必要なファイル

### 1. normal_rand.c

```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

double normal_rand() {
    double u1, u2;
    u1 = (rand() + 1.0) / (RAND_MAX + 2.0);
    u2 = (rand() + 1.0) / (RAND_MAX + 2.0);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

int main(int argc, char *argv[]) {
    int n_samples = atoi(argv[1]);
    srand((unsigned int)time(NULL));
    for (int i = 0; i < n_samples; i++) {
        printf("%.15e\n", normal_rand());
    }
    return 0;
}
```

### 2. brownian_motion.c

```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

double normal_rand() {
    double u1, u2;
    u1 = (rand() + 1.0) / (RAND_MAX + 2.0);
    u2 = (rand() + 1.0) / (RAND_MAX + 2.0);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

int main(int argc, char *argv[]) {
    double gamma = 1.0, kB = 1.0, T = 1.0, m = 1.0, dt = 0.01;
    int n_steps = 1000;
    if (argc >= 2) T = atof(argv[1]);
    if (argc >= 3) m = atof(argv[2]);
    if (argc >= 4) gamma = atof(argv[3]);
    if (argc >= 5) dt = atof(argv[4]);
    if (argc >= 6) n_steps = atoi(argv[5]);
    
    double t = 0.0, rx = 0.0, ry = 0.0, vx = 0.0, vy = 0.0;
    double coeff1 = gamma / m;
    double coeff2 = sqrt(2.0 * gamma * kB * T / m);
    
    srand((unsigned int)time(NULL));
    printf("# t x y vx vy\n");
    printf("%.15e %.15e %.15e %.15e %.15e\n", t, rx, ry, vx, vy);
    
    for (int n = 0; n < n_steps; n++) {
        double eta_x = normal_rand();
        double eta_y = normal_rand();
        vx = vx - coeff1 * vx * dt + coeff2 * sqrt(dt) * eta_x;
        vy = vy - coeff1 * vy * dt + coeff2 * sqrt(dt) * eta_y;
        rx += vx * dt;
        ry += vy * dt;
        t += dt;
        printf("%.15e %.15e %.15e %.15e %.15e\n", t, rx, ry, vx, vy);
    }
    return 0;
}
```

### 3. plot_normal_rand.py

```python
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os

plt.rcParams['font.family'] = 'Hiragino Sans'
plt.rcParams['axes.unicode_minus'] = False

os.makedirs('data', exist_ok=True)
os.makedirs('figures', exist_ok=True)

n_samples_list = [50, 100, 1000]
data_list = []

for n_samples in n_samples_list:
    filename = f'normal_rand_{n_samples}.dat' if n_samples != 1000 else 'normal_rand.dat'
    filepath = os.path.join('data', filename)
    
    with open(filepath, 'w') as f:
        subprocess.run(['./normal_rand', str(n_samples)], stdout=f)
    
    data = np.loadtxt(filepath)
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
```

### 4. visualize_trajectories.py

```python
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
```

### 5. visualize_msd.py

```python
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
msd_individual = []
for _, x, y in trajectories:
    msd_individual.append(x**2 + y**2)

msd = np.array([np.mean([msd_individual[j][i] for j in range(n_runs)]) for i in range(len(t))])

plt.figure(figsize=(10, 8))
colors = plt.cm.tab10(np.linspace(0, 1, n_runs))
for i in range(n_runs):
    plt.plot(t, msd_individual[i], '-', linewidth=1.5, alpha=0.6, 
             color=colors[i], label=f'試行 {i+1}')

plt.plot(t, msd, 'k-', linewidth=3, label='平均 <r²(t)>', alpha=0.9)

plt.xlabel('時間 t')
plt.ylabel('平均二乗変位 <r²(t)>')
plt.title(f'平均二乗変位 (n={n_runs}回実行, T={T}, m={m}, γ={gamma})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join('figures', 'msd_plot.png'), dpi=150)
plt.close()
```

### 6. analyze_diffusion.py

```python
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
```

### 7. analyze_energy.py

```python
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
```

## コンパイル方法

```bash
gcc -o normal_rand normal_rand.c -lm
gcc -o brownian_motion brownian_motion.c -lm
```

## 実行方法

### 課題(1): 正規分布乱数の生成とヒストグラム

```bash
python3 plot_normal_rand.py
```

### 課題(2): ブラウン運動のシミュレーション

```bash
./brownian_motion > data/trajectory.dat
```

### 課題(3): 軌道の可視化

```bash
python3 visualize_trajectories.py 5
```

### 課題(4): 平均二乗変位の計算

```bash
python3 visualize_msd.py 5
```

### 課題(5): 拡散係数の解析（温度・質量・摩擦係数依存性）

```bash
python3 analyze_diffusion.py
```

### 課題(6): エネルギー分布関数

```bash
python3 analyze_energy.py
```
