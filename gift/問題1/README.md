# 問題1 回答

ランジュバン方程式に基づく2次元ブラウン運動のシミュレーションと解析。

## ファイル一覧

| ファイル | 内容 |
|----------|------|
| `Question1.pdf` | 問題文 |
| `langevin.c` | (2) 軌道計算のCプログラム |
| `q1_normal_rand.py` | (1) 正規乱数生成・ヒストグラム |
| `q3_trajectories.py` | (3)(4) 5回実行・軌道の2種類の図 |
| `q5_msd.py` | (5) 平均二乗変位 ⟨r²(t)⟩ のプロット |
| `q6_compare_D.py` | (6) 拡散係数 D の理論とシミュレーションの比較 |
| `report.tex` | (6) 理論（D = k_B T/γ）の導出とレポート |

## 実行方法

```bash
# C のコンパイル
gcc -O2 -o langevin langevin.c -lm

# (1) 正規乱数 → normal_rand.dat, fig_normal_hist.png
python3 q1_normal_rand.py

# (3)(4) 軌道図 → fig_trajectories_xy.png, fig_trajectories_t.png
python3 q3_trajectories.py

# (5) MSD → fig_msd.png
python3 q5_msd.py

# (6) D の比較 → fig_D_compare.png
python3 q6_compare_D.py
```

レポート PDF の作成（TeX が入っている場合）:

```bash
platex report.tex && dvipdfmx report.dvi
# または
uplatex report.tex && dvipdfmx report.dvi
```

## 理論（問題(6)）

拡散係数は **D = k_B T / γ**（質量 m に依存しない）。  
2次元では長時間で ⟨r²(t)⟩ = 4 D t。
