"""
plot_normal_rand.py

目的: 正規分布乱数の生成と可視化
- 異なるサンプル数（50, 100, 1000）で正規分布乱数を生成
- 各サンプル数のヒストグラムを描画し、理論的な正規分布と比較

処理の流れ:
1. Cプログラム（normal_rand.c）をコンパイル（必要に応じて）
2. Cプログラム（normal_rand）を実行して正規分布乱数を生成
3. 生成されたデータを読み込み
4. ヒストグラムと理論曲線を描画して比較
5. 図をファイルに保存
"""

import numpy as np          # 数値計算ライブラリ
import matplotlib.pyplot as plt  # グラフ描画ライブラリ
import subprocess           # 外部プログラム実行用
import os                   # ファイル操作用
import sys                  # システム関連（エラー処理用）

# 日本語フォントの設定（Hiragino Sansを使用）
plt.rcParams['font.family'] = 'Hiragino Sans'
# マイナス記号の文字化けを防ぐ設定
plt.rcParams['axes.unicode_minus'] = False

def compile_c_program(source_file, executable_name):
    """
    Cプログラムをコンパイルする関数
    
    @param source_file: ソースファイルのパス
    @param executable_name: 実行可能ファイルの名前
    @return: コンパイル成功時はTrue、失敗時はFalse
    """
    # ソースファイルの存在確認
    if not os.path.exists(source_file):
        print(f"エラー: ソースファイル '{source_file}' が見つかりません。", file=sys.stderr)
        return False
    
    # 実行可能ファイルが既に存在し、ソースファイルより新しい場合はスキップ
    if os.path.exists(executable_name):
        source_mtime = os.path.getmtime(source_file)
        exec_mtime = os.path.getmtime(executable_name)
        if exec_mtime > source_mtime:
            print(f"実行可能ファイル '{executable_name}' は既に最新です。")
            return True
    
    # コンパイルコマンドを実行
    print(f"コンパイル中: {source_file} -> {executable_name}")
    result = subprocess.run(
        ['gcc', '-o', executable_name, source_file, '-lm'],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"コンパイルエラー:", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        return False
    
    print(f"コンパイル成功: {executable_name}")
    return True

def run_normal_rand(n_samples, executable='./normal_rand'):
    """
    normal_randプログラムを実行して正規分布乱数を生成する関数
    
    @param n_samples: 生成する乱数の個数
    @param executable: 実行可能ファイルのパス
    @return: 生成された乱数の配列、エラー時はNone
    """
    try:
        # プログラムを実行して結果を取得
        result = subprocess.run(
            [executable, str(n_samples)],
            capture_output=True,
            text=True,
            check=True
        )
        
        # 出力を数値のリストに変換
        data = [float(line.strip()) for line in result.stdout.strip().split('\n') if line.strip()]
        
        if len(data) != n_samples:
            print(f"警告: 期待される個数 {n_samples} に対して、実際には {len(data)} 個のデータが生成されました。", file=sys.stderr)
        
        return np.array(data)
    
    except subprocess.CalledProcessError as e:
        print(f"エラー: normal_randの実行に失敗しました。", file=sys.stderr)
        print(f"エラーメッセージ: {e.stderr}", file=sys.stderr)
        return None
    except FileNotFoundError:
        print(f"エラー: 実行可能ファイル '{executable}' が見つかりません。", file=sys.stderr)
        return None
    except Exception as e:
        print(f"エラー: 予期しないエラーが発生しました: {e}", file=sys.stderr)
        return None

# スクリプトのディレクトリに移動（相対パスを正しく解決するため）
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Cプログラムをコンパイル
source_file = 'normal_rand.c'
executable_name = './normal_rand'

if not compile_c_program(source_file, executable_name):
    print("エラー: Cプログラムのコンパイルに失敗しました。", file=sys.stderr)
    sys.exit(1)

# データと図を保存するディレクトリを作成（既に存在する場合は何もしない）
os.makedirs('data', exist_ok=True)      # データファイル用ディレクトリ
os.makedirs('figures', exist_ok=True)   # 図ファイル用ディレクトリ

# 生成する乱数のサンプル数のリスト（50, 100, 1000の3通り）
n_samples_list = [50, 100, 1000]
# 各サンプル数に対応するデータを格納するリスト
data_list = []

# 各サンプル数について正規分布乱数を生成
for n_samples in n_samples_list:
    print(f"処理中: {n_samples}個の正規分布乱数を生成...")
    
    # Cプログラムを実行して正規分布乱数を生成
    data = run_normal_rand(n_samples, executable_name)
    
    if data is None:
        print(f"エラー: {n_samples}個の乱数生成に失敗しました。", file=sys.stderr)
        sys.exit(1)
    
    # ファイル名を決定（1000の場合はnormal_rand.dat、それ以外はnormal_rand_{n_samples}.dat）
    filename = f'normal_rand_{n_samples}.dat' if n_samples != 1000 else 'normal_rand.dat'
    # データディレクトリ内のファイルパスを作成
    filepath = os.path.join('data', filename)
    
    # データをファイルに保存（後で確認できるように）
    np.savetxt(filepath, data, fmt='%.15e')
    print(f"データを保存: {filepath}")
    
    # サンプル数とデータのペアをリストに追加
    data_list.append((n_samples, data))

# 3つのサブプロットを持つ図を作成（横に3つ並べる、サイズは18x5インチ）
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 各サンプル数についてヒストグラムと理論曲線を描画
for idx, (n_samples, data) in enumerate(data_list):
    # ヒストグラムを描画（ビン数20、密度正規化、透明度0.7、黒い枠線）
    axes[idx].hist(data, bins=20, density=True, alpha=0.7, edgecolor='black')
    
    # 理論的な正規分布曲線を描画するためのx座標を生成（データの最小値から最大値まで1000点）
    x = np.linspace(data.min(), data.max(), 1000)
    # 標準正規分布N(0,1)の確率密度関数: (1/√(2π)) * exp(-x²/2)
    theoretical = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * x**2)
    
    # 理論曲線を赤い実線で描画
    axes[idx].plot(x, theoretical, 'r-', linewidth=2, label='理論値 N(0,1)')
    
    # 軸ラベルとタイトルを設定
    axes[idx].set_xlabel('値')              # x軸ラベル
    axes[idx].set_ylabel('確率密度')        # y軸ラベル
    axes[idx].set_title(f'正規分布乱数のヒストグラム (n={n_samples})')  # タイトル
    axes[idx].legend()                      # 凡例を表示
    axes[idx].grid(True, alpha=0.3)         # グリッドを表示（透明度0.3）

# レイアウトを調整して重なりを防ぐ
plt.tight_layout()
# 図をファイルに保存（解像度150dpi）
plt.savefig(os.path.join('figures', 'normal_rand_hist_all.png'), dpi=150)
# メモリを解放するために図を閉じる
plt.close()
