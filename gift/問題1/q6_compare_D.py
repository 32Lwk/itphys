"""
問題1 (6): 拡散係数 D の理論値とシミュレーションの比較
理論: D = kB*T/γ（長時間極限）
シミュレーション: ⟨r^2(t)⟩ ≈ 2D*t から D を推定（30 時間点などで評価）
"""

import numpy as np
import matplotlib.pyplot as plt

from _font_setup import setup_japanese_font
setup_japanese_font()

def run_langevin(seed, n_steps=1000, dt=0.01, gamma=1.0, kB=1.0, T=1.0, m=1.0):
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
    n_runs = 30  # 30 回実行して平均
    n_steps, dt = 1000, 0.01
    gamma, kB, T, m = 1.0, 1.0, 1.0, 1.0
    D_theory = kB * T / gamma  # 理論の拡散係数

    trajectories = [run_langevin(i, n_steps=n_steps, dt=dt, gamma=gamma, kB=kB, T=T, m=m) for i in range(n_runs)]
    t = trajectories[0][0]
    msd = np.mean([trajectories[i][1] for i in range(n_runs)], axis=0)

    # 長時間域で ⟨r^2⟩ = 2Dt にフィット（後半の点で直線回帰）
    # 2次元なので ⟨r^2⟩ = 2 * 2D*t = 4D*t ではなく、⟨|r|^2⟩ = ⟨x^2+y^2⟩ = 2 * (2D_1d * t) = 4 D_1d t.
    # 1次元の拡散係数 D_1d とすると ⟨x^2⟩ = 2 D_1d t。2次元では ⟨r^2⟩ = ⟨x^2⟩+⟨y^2⟩ = 4 D_1d t。
    # 通常 D は1次元あたりの拡散係数なので ⟨x^2⟩ = 2Dt, ⟨r^2⟩ = 4Dt (2次元). 問題では ⟨r^2⟩ なので 4D が傾き。
    # つまり 傾き = 4D => D_sim = 傾き/4.
    use_from = 200  # 十分緩和した後
    t_fit = t[use_from:]
    msd_fit = msd[use_from:]
    A = np.vstack([t_fit, np.ones_like(t_fit)]).T
    slope, _ = np.linalg.lstsq(A, msd_fit, rcond=None)[0]
    D_sim = slope / 4.0  # 2次元: ⟨r^2⟩ = 4 D t

    print("理論 D = kB*T/γ =", D_theory)
    print("シミュレーションから推定 D (⟨r^2⟩=4Dt の傾き/4) =", D_sim)
    print("相対誤差 =", abs(D_sim - D_theory) / D_theory * 100, "%")

    plt.figure(figsize=(6, 4))
    plt.plot(t, msd, "b-", alpha=0.8, label=r"$\langle r^2(t) \rangle$ (30回平均)")
    plt.plot(t, 4 * D_theory * t, "k--", lw=2, label=r"理論 $4Dt$ ($D=k_B T/\gamma$)")
    plt.plot(t_fit, 4 * D_sim * t_fit, "r:", lw=2, label=r"フィット $4 D_{sim} t$")
    plt.xlabel("t")
    plt.ylabel(r"$\langle r^2(t) \rangle$")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("fig_D_compare.png", dpi=150)
    plt.close()
    print("fig_D_compare.png を保存しました。")

if __name__ == "__main__":
    main()
