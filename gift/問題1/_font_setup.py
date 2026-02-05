# 日本語フォント設定（文字化け防止）。各スクリプトで import する
import matplotlib
import matplotlib.font_manager as fm

def setup_japanese_font():
    # 既存の ttflist から Hiragino Sans のパスを1つ取得して登録し直す（名前解決を確実にする）
    for f in fm.fontManager.ttflist:
        p = getattr(f, "fname", "")
        n = getattr(f, "name", "")
        if "Hiragino Sans" in str(n) and "角" in str(p):
            try:
                fm.fontManager.addfont(p)
                prop = fm.FontProperties(fname=p)
                matplotlib.rcParams["font.family"] = prop.get_name()
                matplotlib.rcParams["axes.unicode_minus"] = False
                return
            except Exception:
                pass
    matplotlib.rcParams["font.family"] = ["Hiragino Sans", "sans-serif"]
    matplotlib.rcParams["axes.unicode_minus"] = False
