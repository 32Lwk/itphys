# 問題2: 2準位系のボルツマン統計と温度の数値計算（表1〜5）
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 日本語フォント設定（文字化け防止）
def _setup_japanese_font():
    for f in fm.fontManager.ttflist:
        p, n = getattr(f, "fname", ""), getattr(f, "name", "")
        if "Hiragino Sans" in str(n) and "角" in str(p):
            try:
                fm.fontManager.addfont(p)
                plt.rcParams["font.family"] = fm.FontProperties(fname=p).get_name()
                break
            except Exception:
                pass
    else:
        plt.rcParams["font.family"] = ["Hiragino Sans", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False

_setup_japanese_font()

# ln C(N,M) = Σ_{k=1}^M ln(N-k+1) - Σ_{k=1}^M ln(k)
def entropy_boltzmann(N, M):
    if M == 0:
        return 0.0
    k = np.arange(1, M + 1, dtype=float)
    return np.sum(np.log(N - k + 1)) - np.sum(np.log(k))

def main():
    N_list = [20, 50, 100]

    # 表1: 各 N について全 M で S(M) を計算
    for N in N_list:
        S_list = [entropy_boltzmann(N, M) for M in range(N + 1)]
        # 表1の結果（必要に応じてコメントアウト解除）
        # for M in range(N + 1): print(f"N={N} M={M} S={S_list[M]:.6f}")

    # 表2・表3: S/N vs x = M/N のプロット（Stirling 比較含む）
    plt.figure(figsize=(6, 4))
    x_cont = np.linspace(1e-6, 1 - 1e-6, 200)
    s_stirling = -(x_cont * np.log(x_cont) + (1 - x_cont) * np.log(1 - x_cont))
    plt.plot(x_cont, s_stirling, 'k-', lw=2, label='Stirling (N→∞)')

    for N in N_list:
        M_arr = np.arange(N + 1)
        x = M_arr / N
        S = np.array([entropy_boltzmann(N, m) for m in M_arr])
        s = S / N
        plt.plot(x, s, 'o', ms=3, label=f'N={N}')
    plt.xlabel(r'$x = E/E_0 = M/N$')
    plt.ylabel(r'$s = S/N$')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig_entropy.png', dpi=150)
    plt.close()
    print('fig_entropy.png を保存しました。')

    # 表4: 温度の数値計算 T(M) ≈ (S(M+1)-S(M-1)) / (E(M+1)-E(M-1)), E=M (ε=1)
    # 表5: T のプロット（Stirling 理論値との比較）
    # 理論曲線は x=0.5 で発散するため y をクリップして軸スケールを抑え、全 N の点が見えるようにする
    plt.figure(figsize=(6, 4))
    x_cont = np.linspace(1e-6, 1 - 1e-6, 200)
    T_th = 1.0 / np.log((1 - x_cont) / x_cont)
    y_clip = 8
    T_th_clip = np.clip(T_th, -y_clip, y_clip)
    plt.plot(x_cont, T_th_clip, 'k-', lw=2, label=r'Stirling $T_{\mathrm{th}}(x)$')

    markers = ['o', 's', '^']
    for i, N in enumerate(N_list):
        x_plot, T_plot = [], []
        for M in range(1, N):
            dS = entropy_boltzmann(N, M + 1) - entropy_boltzmann(N, M - 1)
            if abs(dS) < 1e-12:
                continue
            T_plot.append(2.0 / dS)
            x_plot.append(M / N)
        mk = markers[i % len(markers)]
        plt.plot(x_plot, T_plot, marker=mk, ls='', ms=4, label=f'N={N}')
    plt.ylim(-y_clip, y_clip)
    plt.xlabel(r'$x = E/E_0 = M/N$')
    plt.ylabel(r'$T$')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig_temperature.png', dpi=150)
    plt.close()
    print('fig_temperature.png を保存しました。')

if __name__ == '__main__':
    main()
