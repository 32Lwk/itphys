"""
問題1 (3)(4): Cプログラムを5回実行し、軌道を2種類の図で可視化
- 図1: (x, y) の2次元軌道 5本
- 図2: 速度 (vx, vy) または時刻に対する x, y など（2つ目の図）
"""

import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os

from _font_setup import setup_japanese_font
setup_japanese_font()

def run_langevin_c(seed):
    """langevin.c を実行し、データを返す。seed=0 のときはC側の乱数に任せる。"""
    cmd = ["./langevin", str(seed)] if seed != 0 else ["./langevin"]
    try:
        out = subprocess.check_output(cmd, cwd=os.path.dirname(os.path.abspath(__file__)), text=True)
    except FileNotFoundError:
        # C が未コンパイルの場合は Python で同じシミュレーション
        return run_langevin_python(seed)
    lines = [L for L in out.strip().split("\n") if not L.startswith("#")]
    data = np.loadtxt(lines)
    t = data[:, 0]
    x, y = data[:, 1], data[:, 2]
    vx, vy = data[:, 3], data[:, 4]
    return t, x, y, vx, vy

def run_langevin_python(seed):
    """C と同一のスキームで Python シミュレーション（γ=kB=T=m=1, dt=0.01, 1000 steps）"""
    np.random.seed(seed if seed != 0 else None)
    gamma, kB, T, m = 1.0, 1.0, 1.0, 1.0
    dt, n_steps = 0.01, 1000
    c1, c2 = gamma / m, np.sqrt(2 * gamma * kB * T / m) * np.sqrt(dt)
    x, y = 0.0, 0.0
    vx, vy = 0.0, 0.0
    t = [0.0]
    X, Y = [x], [y]
    VX, VY = [vx], [vy]
    for n in range(n_steps):
        eta_x, eta_y = np.random.normal(0, 1), np.random.normal(0, 1)
        vx = vx - c1 * vx * dt + c2 * eta_x
        vy = vy - c1 * vy * dt + c2 * eta_y
        x += vx * dt
        y += vy * dt
        t.append((n + 1) * dt)
        X.append(x); Y.append(y)
        VX.append(vx); VY.append(vy)
    return np.array(t), np.array(X), np.array(Y), np.array(VX), np.array(VY)

def main():
    base = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base)

    trajectories = []
    for i in range(5):
        seed = 100 + i  # 5 通り
        t, x, y, vx, vy = run_langevin_c(seed)
        trajectories.append((t, x, y, vx, vy))

    # 図1: (x, y) 軌道 5 本
    plt.figure(figsize=(6, 5))
    for t, x, y, vx, vy in trajectories:
        plt.plot(x, y, alpha=0.8)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("2次元軌道（5回実行）")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("fig_trajectories_xy.png", dpi=150)
    plt.close()

    # 図2: 時刻に対する位置 x(t), y(t) の5本
    plt.figure(figsize=(7, 4))
    for t, x, y, vx, vy in trajectories:
        plt.plot(t, x, alpha=0.7)
        plt.plot(t, y, alpha=0.7, linestyle="--")
    plt.xlabel("t")
    plt.ylabel("x, y")
    plt.title("位置の時系列（5回実行・実線: x, 破線: y）")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("fig_trajectories_t.png", dpi=150)
    plt.close()

    print("fig_trajectories_xy.png と fig_trajectories_t.png を保存しました。")

if __name__ == "__main__":
    main()
