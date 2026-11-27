"""YOLO11n vs YOLO11s 비교 (B만). E1과 같은 방식으로 평가해 results/yolo_s/에 저장.

n = candidates/candidates_avg_B.csv (E1에 쓴 후보), s = candidates/candidates_avg_s_B.csv
- 임계값 0.001~0.95 (0.001 간격) 스윕 -> FPPI 0.25/0.5/1.0 Recall (전체/Q1/겹침, 선형 보간)
- conf 0.25: FPPI, Recall, 미탐 원인 (hit / low_conf / no_candidate)
- n 수치는 results/E1/ 의 기존 csv와 일치하는지 확인한다 (기존 파일은 읽기만)

python code/run_yolo_s.py
"""
import json
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
OUT = os.path.join(E.RESULTS_DIR, "yolo_s")
MODELS = {"YOLO11n": "avg_B", "YOLO11s": "avg_s_B"}
COLORS = {"YOLO11n": "#2a6fdb", "YOLO11s": "#d9480f"}
THRS = np.round(np.arange(0.001, 0.9505, 0.001), 3)
GROUPS = [("recall", "전체"), ("recall_q1", "Q1 (높이 < 184px)"), ("recall_overlap", "겹침 (IoA > 0.3)")]


def read_result(path):
    with open(path, encoding="utf-8") as f:
        first = f.readline()
    return pd.read_csv(path, skiprows=0 if "," in first else 1)


def main():
    os.makedirs(OUT, exist_ok=True)
    gts = E.load_gt("B")
    curves, rows, causes = {}, [], []
    for name, tag in MODELS.items():
        cands = E.load_candidates(os.path.join(E.ROOT, "candidates", f"candidates_{tag}.csv"))
        cur = E.sweep(cands, gts, THRS)
        curves[name] = cur
        E.save_csv(cur, os.path.join(OUT, f"curve_{tag}.csv"), "")
        r25 = E.fppi_recall(cands, gts, 0.25)
        mc = E.miss_cause_table(E.miss_causes(cands, gts, 0.25))
        E.save_csv(mc, os.path.join(OUT, f"miss_causes_{tag}.csv"), "")
        row = {"model": name, "candidates": f"candidates_{tag}.csv", "n_cands": len(cands),
               "cands_per_image_max": int(cands.groupby("image_id").size().max())}
        for f in (0.25, 0.5, 1.0):
            for col, _ in GROUPS:
                row[f"{col}@fppi{f}"] = E.recall_at_fppi(cur, f, col)
        row.update({"conf0.25_fppi": r25["fppi"], "conf0.25_recall": r25["recall"],
                    "conf0.25_recall_q1": r25["recall_q1"], "conf0.25_recall_overlap": r25["recall_overlap"]})
        rows.append(row)
        for _, m in mc.iterrows():
            causes.append({"model": name, **m.to_dict()})
        E.save_csv(pd.DataFrame([{"fppi": f, **{k: E.recall_at_fppi(cur, f, c) for c, k in
                                                 [("recall", "all"), ("recall_q1", "q1"), ("recall_overlap", "overlap")]}}
                                 for f in (0.25, 0.5, 1.0)]), os.path.join(OUT, f"recall_at_fppi_{tag}.csv"), "")

    # n이 기존 E1 결과와 같은지 확인
    e1 = read_result(os.path.join(E.RESULTS_DIR, "E1", "E1_recall_at_fppi_avg_B.csv")).set_index("fppi")
    n = rows[0]
    for f in (0.25, 0.5, 1.0):
        for col, k in [("recall", "all"), ("recall_q1", "q1"), ("recall_overlap", "overlap")]:
            assert abs(n[f"{col}@fppi{f}"] - e1.loc[f, k]) < 5e-5, (f, k, n[f"{col}@fppi{f}"], e1.loc[f, k])

    comp = pd.DataFrame(rows)
    diff = {"model": "s − n (%p)"}
    for c in comp.columns[4:]:
        diff[c] = 100 * (comp[c].iloc[1] - comp[c].iloc[0]) if "fppi" not in c or "@" in c else comp[c].iloc[1] - comp[c].iloc[0]
    comp = pd.concat([comp, pd.DataFrame([diff])], ignore_index=True)
    E.save_csv(comp, os.path.join(OUT, "compare_n_vs_s_B.csv"), "",
               "B, E1과 같은 방식. n = 기존 E1 후보. 마지막 행: Recall은 %p 차이, conf0.25_fppi는 그대로 차이")
    mc = pd.DataFrame(causes)
    E.save_csv(mc, os.path.join(OUT, "miss_causes_n_vs_s_B.csv"), "", "conf 0.25 기준 미탐 원인 비율")

    # 그래프 1: FPPI-Recall (3칸)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharey=False)
    for ax, (col, label) in zip(axes, GROUPS):
        lo = 1.0
        for name, cur in curves.items():
            c = cur[cur.fppi <= 2.0].sort_values("fppi")
            ax.plot(c.fppi, c[col], lw=2, color=COLORS[name], label=name)
            p = cur.loc[np.isclose(cur.thr, 0.25)].iloc[0]
            ax.plot(p.fppi, p[col], "o", color=COLORS[name], ms=5)
            lo = min(lo, c[col].min())
        for f in (0.25, 0.5, 1.0):
            ax.axvline(f, color="#bbb", lw=0.8, ls=":")
        ax.set_xlim(0, 2)
        ax.set_ylim(max(0, lo - 0.05), 1.0)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("FPPI (이미지당 오탐 수)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Recall")
    axes[0].legend(fontsize=9, title="점 = conf 0.25", title_fontsize=8)
    fig.suptitle("YOLO11n vs YOLO11s: FPPI-Recall 곡선  [avg, B]", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fppi_recall_n_vs_s_B.png"), dpi=150)
    plt.close(fig)

    # 그래프 2: 미탐 원인 (conf 0.25) 가로 누적 막대
    names = {"all": "전체", "q1": "Q1", "overlap": "겹침"}
    parts = [("hit", "hit", "#2a6fdb"), ("low_conf", "low_conf", "#d9480f"), ("no_candidate", "no_candidate", "#868e96")]
    order = [(g, m) for g in ("all", "q1", "overlap") for m in MODELS][::-1]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    y = np.arange(len(order))
    left = np.zeros(len(order))
    for col, label, color in parts:
        v = np.array([mc[(mc.group == g) & (mc.model == m)][col].iloc[0] for g, m in order])
        ax.barh(y, v, left=left, color=color, height=0.65, label=label, edgecolor="white")
        for i, (l, w) in enumerate(zip(left, v)):
            if w >= 0.06:
                ax.text(l + w / 2, y[i], f"{100 * w:.1f}%", ha="center", va="center", color="white", fontsize=9,
                        fontweight="bold")
            else:
                ax.text(1.01, y[i], f"{100 * w:.1f}%", ha="left", va="center", color=color, fontsize=9,
                        fontweight="bold")
        left += v
    ax.set_yticks(y, [f"{names[g]} · {m}" for g, m in order])
    ax.set_xlim(0, 1.09)
    ax.set_xticks(np.linspace(0, 1, 6), [f"{int(100 * t)}%" for t in np.linspace(0, 1, 6)])
    ax.set_xlabel("정답 중 비율")
    ax.set_title("미탐 원인 (conf 0.25 기준): YOLO11n vs YOLO11s  [avg, B]", fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "miss_causes_n_vs_s_B.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    pd.set_option("display.width", 220)
    print(comp.T.to_string())
    print(mc.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
