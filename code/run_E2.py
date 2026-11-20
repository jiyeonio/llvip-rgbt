"""E2: 점수 치우침 (reliability diagram).

라벨: 전체 후보(conf >= 0.001)를 conf 높은 순 1:1 매칭(IoU >= 0.5). 짝 있으면 1.
그룹(후보 박스 기준): 크기 = 높이 184px 미만 작음 / 보통,
                     겹침 = 다른 후보(conf >= 0.1)와 IoA > 0.3
python code/run_E2.py candidates/candidates_proxy_test.csv
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

BINS = [0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, np.inf]
BIN_LABELS = ["0.001-0.01", "0.01-0.05", "0.05-0.1", "0.1-0.15", "0.15-0.2",
              "0.2-0.25", "0.25-0.3", "0.3-0.4", "0.4-0.5", ">=0.5"]
MIN_N = 50
NAME_KO = {"small": "작은 후보", "normal": "보통 후보", "overlap": "겹친 후보", "no_overlap": "안 겹친 후보"}
OV_REF_CONF = 0.1


def candidate_labels(cands, gts):
    """E2 라벨 + 그룹 열을 붙인 후보 DataFrame (E3에서도 재사용)."""
    c = cands[cands.conf >= 0.001].copy()
    m = E.match(c, gts, 0.001)
    c["label"] = m["cand_tp"].astype(int)
    c["gt_idx"] = m["cand_gt"]                          # 짝 정답의 gts 인덱스 (-1 = 없음), E3용
    c["height"] = c.y2 - c.y1
    c["small"] = c.height < E.Q1_HEIGHT
    ov = np.zeros(len(c), bool)
    box = c[["x1", "y1", "x2", "y2"]].to_numpy(float)
    conf = c.conf.to_numpy()
    for _, pos in c.groupby("image_id").indices.items():
        ref = pos[conf[pos] >= OV_REF_CONF]
        if len(ref) == 0:
            continue
        m = E.ioa_matrix(box[pos], box[ref])
        m[pos[:, None] == ref[None, :]] = 0.0          # 자기 자신 제외
        ov[pos] = (m > E.OVERLAP_IOA).any(axis=1)
    c["overlap"] = ov
    c["bin"] = pd.cut(c.conf, BINS, right=False, labels=BIN_LABELS)
    return c


def table(c):
    rows = []
    groups = [("size", "small", c.small), ("size", "normal", ~c.small),
              ("overlap", "overlap", c.overlap), ("overlap", "no_overlap", ~c.overlap),
              ("all", "all", pd.Series(True, index=c.index))]
    for gtype, gname, mask in groups:
        sub = c[mask.to_numpy()]
        g = sub.groupby("bin", observed=False)
        for b in BIN_LABELS:
            s = g.get_group(b) if b in g.groups and len(g.get_group(b)) else sub.iloc[:0]
            rows.append(dict(group_type=gtype, group=gname, bin=b, n=len(s),
                             person_ratio=s.label.mean() if len(s) else np.nan,
                             mean_conf=s.conf.mean() if len(s) else np.nan,
                             low_n=len(s) < MIN_N))
    return pd.DataFrame(rows)


def plot(tab, gtype, names, colors, title, path, note):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.arange(len(BIN_LABELS))
    allr = tab[tab.group == "all"].set_index("bin").loc[BIN_LABELS]
    ax.plot(x, allr.mean_conf, color="#999", ls=":", lw=1.2, label="구간 평균 conf (점수가 정확하면 이 선 위)")
    for name, col in zip(names, colors):
        t = tab[(tab.group_type == gtype) & (tab.group == name)].set_index("bin").loc[BIN_LABELS]
        y, low = t.person_ratio.to_numpy(float), t.low_n.to_numpy()
        ax.plot(x, y, color=col, lw=2, label=NAME_KO.get(name, name))
        # n < 50 구간: 점선 연결 + 속 빈 마커
        for i in range(len(x) - 1):
            if low[i] or low[i + 1]:
                ax.plot(x[i:i + 2], y[i:i + 2], color="white", lw=2.6)
                ax.plot(x[i:i + 2], y[i:i + 2], color=col, lw=1.6, ls="--")
        ax.scatter(x[~low], y[~low], color=col, zorder=3, s=28)
        ax.scatter(x[low], y[low], facecolors="white", edgecolors=col, zorder=3, s=28)
    ax.set_xticks(x, [b.replace(">=", "≥") for b in BIN_LABELS], rotation=35, ha="right")
    ax.set_ylim(0, 1)
    ax.set_xlabel("conf 구간")
    ax.set_ylabel("사람 비율 (정답과 짝지어진 후보, IoU ≥ 0.5)")
    ax.set_title(title + ("\n" + note if note else ""), fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left", title=f"속 빈 점·점선: 후보 {MIN_N}개 미만", title_fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(E.ROOT, "candidates", "candidates_proxy_test.csv")
    model, split = E.parse_cand_name(path)
    tag = f"{model}_{split}"
    out = os.path.join(E.RESULTS_DIR, "E2")
    os.makedirs(out, exist_ok=True)

    c = candidate_labels(E.load_candidates(path), E.load_gt(split))
    tab = table(c)
    E.save_csv(tab, os.path.join(out, f"E2_reliability_{tag}.csv"), model)
    note = E.PROXY_NOTE_PNG if model == "proxy" else ""
    plot(tab, "size", ["small", "normal"], ["#d9480f", "#2a6fdb"],
         f"E2 점수 치우침: 크기별 (작음 = 높이 184px 미만)  [{tag}]",
         os.path.join(out, f"E2_reliability_size_{tag}.png"), note)
    plot(tab, "overlap", ["overlap", "no_overlap"], ["#d9480f", "#2a6fdb"],
         f"E2 점수 치우침: 겹침별 (conf ≥ 0.1 후보와 IoA > 0.3)  [{tag}]",
         os.path.join(out, f"E2_reliability_overlap_{tag}.png"), note)
    pd.set_option("display.width", 160)
    print(f"candidates={len(c)} positives={c.label.sum()} small={c.small.mean():.3f} overlap={c.overlap.mean():.3f}")
    print(tab.pivot(index="bin", columns="group", values="person_ratio").loc[BIN_LABELS].round(3))
    print(tab.pivot(index="bin", columns="group", values="n").loc[BIN_LABELS])
