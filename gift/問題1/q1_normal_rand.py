"""
問題1 (1): 正規分布に従う乱数の生成とヒストグラム
50, 100, 1000 個の正規乱数を生成し normal_rand.dat に保存。
ヒストグラム（bins=20）を表示する。
Python 3 + matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from _font_setup import setup_japanese_font
setup_japanese_font()

# 正規乱数 1000 個を生成して normal_rand.dat に保存（50, 100 も必要なら n を変えて実行）
np.random.seed(42)
data = np.random.normal(0, 1, 1000)
np.savetxt("normal_rand.dat", data)

plt.figure(figsize=(6, 4))
plt.hist(data, bins=20, density=True, alpha=0.7, edgecolor="black", label="ヒストグラム")
# 理論曲線
x = np.linspace(-4, 4, 200)
plt.plot(x, np.exp(-x**2/2)/np.sqrt(2*np.pi), "r-", lw=2, label="理論 N(0,1)")
plt.xlabel("x")
plt.ylabel("密度")
plt.legend()
plt.title("正規分布乱数 (n=1000, bins=20)")
plt.tight_layout()
plt.savefig("fig_normal_hist.png", dpi=150)
plt.show()
