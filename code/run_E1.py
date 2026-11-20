"""E1: FPPI-Recall 곡선 + FPPI별 Recall 표 + 미탐 원인 표.

python code/run_E1.py candidates/candidates_proxy_test.csv
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

path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(E.ROOT, "candidates", "candidates_proxy_test.csv")
model, split = E.parse_cand_name(path)
tag = f"{model}_{split}"
out = os.path.join(E.RESULTS_DIR, "E1")
os.makedirs(out, exist_ok=True)

gts = E.load_gt(split)
cands = E.load_candidates(path)

# 임계값: 0.001 ~ 0.95, 0.001 간격 (950점)
thrs = np.round(np.arange(0.001, 0.9505, 0.001), 3)
curve = E.sweep(cands, gts, thrs)
# 스윕 결과를 직접 매칭과 대조 (몇 점)
for t in (0.001, 0.1, 0.25, 0.5, 0.9):
    d = E.fppi_recall(cands, gts, t)
    r = curve.loc[np.isclose(curve.thr, t)].iloc[0]
    assert d["tp"] == r.tp and d["fp"] == r.fp and abs(d["recall_q1"] - r.recall_q1) < 1e-12, (t, d, r)
E.save_csv(curve, os.path.join(out, f"E1_curve_{tag}.csv"), model)

groups = [("recall", "전체", "all"), ("recall_q1", "Q1 (작은 사람, 높이 184px 미만)", "q1"),
          ("recall_overlap", "겹친 사람 (IoA > 0.3)", "overlap")]

# 그래프 3장
for col, label, key in groups:
    c = curve[curve.fppi <= 2.0].sort_values("fppi")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(c.fppi, c[col], lw=2, color="#2a6fdb")
    p = curve.loc[np.isclose(curve.thr, 0.25)].iloc[0]
    ax.plot(p.fppi, p[col], "o", color="#d9480f")
    ax.annotate(f"conf 0.25\nFPPI {p.fppi:.2f}, Recall {p[col]:.3f}", (p.fppi, p[col]),
                textcoords="offset points", xytext=(8, -28), fontsize=9)
    for f in (0.25, 0.5, 1.0):
        ax.axvline(f, color="#bbb", lw=0.8, ls=":")
    ax.set_xlim(0, 2)
    ax.set_ylim(max(0, c[col].min() - 0.05), 1.0)
    ax.set_xlabel("FPPI (이미지당 오탐 수)")
    ax.set_ylabel("Recall (찾은 사람 비율)")
    ttl = f"E1 FPPI-Recall 곡선: {label}  [{tag}]"
    if model == "proxy":
        ttl += "\n" + E.PROXY_NOTE_PNG
    ax.set_title(ttl, fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"E1_curve_{key}_{tag}.png"), dpi=150)
    plt.close(fig)

# FPPI 0.25 / 0.5 / 1.0 에서의 Recall
tab = pd.DataFrame([{"fppi": f, **{key: E.recall_at_fppi(curve, f, col) for col, _, key in groups}}
                    for f in (0.25, 0.5, 1.0)])
E.save_csv(tab, os.path.join(out, f"E1_recall_at_fppi_{tag}.csv"), model)

# 미탐 원인 (conf 0.25)
causes = E.miss_causes(cands, gts, 0.25)
mc = E.miss_cause_table(causes)
E.save_csv(mc, os.path.join(out, f"E1_miss_causes_{tag}.csv"), model)

print(tab.round(4).to_string(index=False))
print(mc.round(4).to_string(index=False))
