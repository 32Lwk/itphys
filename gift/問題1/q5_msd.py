"""
問題1 (5): 平均二乗変位 ⟨r^2(t)⟩ = ⟨|r(t) - r(0)|^2⟩ の計算
5 回実行し、⟨r^2(t)⟩ をプロットする。
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from _font_setup import setup_japanese_font
setup_japanese_font()

def run_langevin_python(seed, n_steps=1000, dt=0.01):
    gamma, kB, T, m = 1.0, 1.0, 1.0, 1.0
    np.random.seed(seed)
    c1 = gamma / m
    c2 = np.sqrt(2 * gamma * kB * T / m) * np.sqrt(dt)
    x, y = 0.0, 0.0
    vx, vy = 0.0, 0.0
    r2_list = [0.0]
    for n in range(n_steps):
        eta_x, eta_y = np.random.normal(0, 1), np.random.normal(0, 1)
        vx = vx - c1 * vx * dt + c2 * eta_x
        vy = vy - c1 * vy * dt + c2 * eta_y
        x += vx * dt
        y += vy * dt
        r2_list.append(x**2 + y**2)
    t = np.arange(n_steps + 1) * dt
    return t, np.array(r2_list)

def main():
    n_runs = 5
    trajectories = [run_langevin_python(200 + i) for i in range(n_runs)]
    t = trajectories[0][0]
    # ⟨r^2(t)⟩
    msd = np.mean([trajectories[i][1] for i in range(n_runs)], axis=0)

    plt.figure(figsize=(6, 4))
    plt.plot(t, msd, "b-", lw=2, label=r"$\langle r^2(t) \rangle$ (5回平均)")
    # 理論: 2次元では ⟨r^2⟩ = 4Dt, D = kB*T/γ
    D_theory = 1.0  # kB=T=γ=1
    plt.plot(t, 4 * D_theory * t, "k--", alpha=0.8, label=r"理論 $4Dt$ ($D=k_B T/\gamma$)")
    plt.xlabel("t")
    plt.ylabel(r"$\langle r^2(t) \rangle$")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("fig_msd.png", dpi=150)
    plt.close()
    print("fig_msd.png を保存しました。")

if __name__ == "__main__":
    main()
