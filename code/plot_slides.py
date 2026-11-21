"""발표용 그래프 (기존 결과 csv를 읽기만 한다).

1. results/E1/E1_miss_causes_{tag}.png : 미탐 원인 가로 누적 막대 (conf 0.25)
2. results/E4/E4_0_headroom_{tag}.png  : 실제 Recall vs 개선 여지 상한 묶음 막대 (FPPI 0.5)

python code/plot_slides.py [avg_B]
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval as E  # noqa: E402

E.use_korean_font()
BLUE, ORANGE, GREY = "#2a6fdb", "#d9480f", "#868e96"


def read_result(path):
    """첫 줄이 표시 문구인 csv도 읽는다."""
    with open(path, encoding="utf-8") as f:
        first = f.readline()
    return pd.read_csv(path, skiprows=0 if "," in first else 1)


def miss_causes(tag):
    df = read_result(os.path.join(E.RESULTS_DIR, "E1", f"E1_miss_causes_{tag}.csv"))
    names = {"all": "전체", "q1": "Q1 (작은 사람)", "overlap": "겹친 사람"}
    df = df.set_index("group").loc[["all", "q1", "overlap"]][::-1]       # 위에서부터 전체 / Q1 / 겹침
    parts = [("hit", "hit (conf 0.25 이상으로 찾음)", BLUE),
             ("low_conf", "low_conf (후보는 있으나 점수 부족)", ORANGE),
             ("no_candidate", "no_candidate (후보 없음)", GREY)]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    y = np.arange(len(df))
    left = np.zeros(len(df))
    for col, label, color in parts:
        v = df[col].to_numpy()
        ax.barh(y, v, left=left, color=color, height=0.6, label=label, edgecolor="white", linewidth=1)
        for i, (l, w) in enumerate(zip(left, v)):
            if w >= 0.06:
                ax.text(l + w / 2, y[i], f"{100 * w:.1f}%", ha="center", va="center", color="white",
                        fontsize=10, fontweight="bold")
            else:                                                       # 좁은 구간은 막대 오른쪽 밖에
                ax.text(1.01, y[i], f"{100 * w:.1f}%", ha="left", va="center", color=color,
                        fontsize=10, fontweight="bold")
        left += v
    ax.set_yticks(y, [f"{names[g]}\n(정답 {int(n):,}개)" for g, n in zip(df.index, df.n_gt)])
    ax.set_xlim(0, 1.09)
    ax.set_xticks(np.linspace(0, 1, 6), [f"{int(100 * t)}%" for t in np.linspace(0, 1, 6)])
    ax.set_xlabel("정답 중 비율")
    ax.set_title(f"E1 미탐 원인 (conf 0.25 기준)  [{tag}]", fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = os.path.join(E.RESULTS_DIR, "E1", f"E1_miss_causes_{tag}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def headroom(tag):
    df = read_result(os.path.join(E.RESULTS_DIR, "E4", f"E4_0_upper_bound_{tag}.csv")).set_index("group")
    groups = [("전체", "전체"), ("Q1", "Q1 (작은 사람)"), ("겹침", "겹친 사람")]
    bars = [("recall", "실제 Recall (FPPI 0.5)", BLUE, 1.0),
            ("upper_conf>=0.01", "상한: conf ≥ 0.01 후보만", ORANGE, 0.5),
            ("upper_all_cands", "상한: 저장된 전체 후보", ORANGE, 1.0)]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.arange(len(groups))
    w = 0.26
    for k, (col, label, color, alpha) in enumerate(bars):
        v = np.array([df.loc[g, col] for g, _ in groups])
        b = ax.bar(x + (k - 1) * w, v, w, color=color, alpha=alpha, label=label, edgecolor="white")
        for rect, val in zip(b, v):
            ax.text(rect.get_x() + rect.get_width() / 2, val + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x, [f"{name}\n(정답 {int(df.loc[g, 'n_gt']):,}개)" for g, name in groups])
    ax.set_ylim(0, 1.08)                       # 막대그래프는 0부터 (길이 비교가 왜곡되지 않게)
    ax.set_ylabel("Recall")
    thr, fppi = df.thr.iloc[0], df.fppi.iloc[0]
    ax.set_title(f"E4-0 개선 여지 상한: 놓친 정답 중 IoU ≥ 0.5 후보가 있는 정답을 모두 살린다면  [{tag}]\n"
                 f"원래 점수 단일 합격선 {thr:.3f} (FPPI {fppi:.3f}) 기준", fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = os.path.join(E.RESULTS_DIR, "E4", f"E4_0_headroom_{tag}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "avg_B"
    print(miss_causes(tag))
    print(headroom(tag))
