"""E3: 그룹별 합격선.

**B 후보로만 실행한다.** split이 B가 아닌 후보 파일(예: proxy test)은 거부한다.

그룹 (후보 박스 기준, E2와 같은 정의):
  small        : 높이 184px 미만 (겹쳐도 여기)
  normal_ov    : 보통 & 다른 후보(conf >= 0.1)와 IoA > 0.3
  normal_noov  : 보통 & 안 겹침
합격선: 0.01 ~ 0.60, 0.01 간격 (60^3 = 216,000 조합)
목표: 전체 FPPI <= 0.5 에서 (Q1 Recall + 겹침 Recall) / 2 최대
제약: 전체 Recall >= 같은 FPPI에서 단일 합격선의 전체 Recall
      (단일 합격선 곡선을 0.001 간격으로 훑고, 조합의 FPPI에서 선형 보간한 값과 비교)
동점: 전체 Recall 높은 조합 -> 합격선 합이 큰 조합
빠른 탐색: E2 라벨(conf >= 0.001 전체 1:1 매칭)을 고정하고, 그룹별·합격선별 TP/FP/Q1-TP/겹침-TP
          누적값을 미리 계산해 더한다. 최종 조합만 실제 1:1 매칭으로 다시 계산한다.
          실제 매칭이 FPPI/제약을 어기면 다음 순위 조합으로 넘어간다 (최대 MAX_VERIFY개).

교차검증 모드 (--mode cv, 기본은 both):
  split.csv의 segment_id로 B를 5 fold로 나눔 (같은 segment는 같은 fold, 이미지 수 균형, seed 42).
  fold마다 나머지 4 fold로 조합을 고르고(위와 같은 규칙), 빠진 fold에서
  그룹별 합격선과 단일 합격선(빠진 fold에서 같은 FPPI로 보간)의 Recall(전체/Q1/겹침)·FPPI 계산.
  E2 라벨은 이미지 단위 매칭이라 B 전체에서 한 번 계산해 fold별로 잘라 써도 결과가 같다.
B 전체로 고르는 기존 방식은 "참고용(in-sample)" 표시 (파일 이름 E3_insample_*).

python code/run_E3.py candidates/candidates_avg_B.csv [--mode insample|cv|both]
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval as E  # noqa: E402
from run_E2 import candidate_labels  # noqa: E402

THRS = np.round(np.arange(1, 61) / 100, 2)          # 0.01 ~ 0.60
GROUPS = ["small", "normal_ov", "normal_noov"]
FPPI_MAX = 0.5
MAX_VERIFY = 50
SINGLE_THRS = np.round(np.arange(0.001, 0.9505, 0.001), 3)


def assign_groups(c: pd.DataFrame) -> pd.Series:
    return pd.Series(np.where(c.small, "small", np.where(c.overlap, "normal_ov", "normal_noov")),
                     index=c.index)


def group_tables(c: pd.DataFrame, gts: pd.DataFrame) -> dict:
    """그룹별 (len(THRS),) 배열: tp, fp, tp_q1, tp_ov (conf >= thr 인 후보 기준, E2 라벨)."""
    gq1 = gts["q1"].to_numpy()
    gov = gts["overlap"].to_numpy()
    gpos = gts.index.get_indexer(c["gt_idx"].where(c["gt_idx"] >= 0, gts.index[0]))
    lab = c["label"].to_numpy().astype(bool)
    is_q1 = lab & gq1[gpos]
    is_ov = lab & gov[gpos]
    grp = c["group"].to_numpy()
    conf = c["conf"].to_numpy()
    out = {}
    for g in GROUPS:
        m = grp == g

        def cnt(mask):
            s = np.sort(conf[m & mask])
            return len(s) - np.searchsorted(s, THRS, "left")
        tp = cnt(lab)
        out[g] = dict(tp=tp, fp=cnt(~lab), tp_q1=cnt(is_q1), tp_ov=cnt(is_ov))
    return out


def fast_search(tab: dict, n_img: int, n_gt: int, n_q1: int, n_ov: int, single_curve: pd.DataFrame):
    """216,000 조합을 한 번에 계산해 조건을 만족하는 조합을 순위대로 반환."""
    a, b, d = (tab[g] for g in GROUPS)
    S = lambda k: a[k][:, None, None] + b[k][None, :, None] + d[k][None, None, :]  # noqa: E731
    tp, fp, tq, to = S("tp"), S("fp"), S("tp_q1"), S("tp_ov")
    fppi = fp / n_img
    rec = tp / n_gt
    rq1 = tq / n_q1 if n_q1 else np.full(tp.shape, np.nan)
    rov = to / n_ov if n_ov else np.full(tp.shape, np.nan)
    obj = (rq1 + rov) / 2

    # 단일 합격선 전체 Recall을 조합의 FPPI에서 보간 (FPPI 오름차순 곡선)
    sc = single_curve.sort_values(["fppi", "recall"])
    base = np.interp(fppi, sc.fppi.to_numpy(), sc.recall.to_numpy())
    ok = (fppi <= FPPI_MAX) & (rec >= base - 1e-12)

    i, j, k = np.nonzero(ok)
    res = pd.DataFrame({"t_small": THRS[i], "t_normal_ov": THRS[j], "t_normal_noov": THRS[k],
                        "fppi": fppi[ok], "recall": rec[ok], "recall_q1": rq1[ok],
                        "recall_overlap": rov[ok], "objective": obj[ok], "single_recall_same_fppi": base[ok]})
    res["thr_sum"] = res.t_small + res.t_normal_ov + res.t_normal_noov
    res = res.sort_values(["objective", "recall", "thr_sum"], ascending=False, kind="stable")
    return res.reset_index(drop=True), int(ok.sum())


def real_eval(cands_lab: pd.DataFrame, gts: pd.DataFrame, t: dict, single_curve: pd.DataFrame) -> dict:
    """그룹별 합격선을 적용한 후보로 실제 1:1 매칭."""
    thr = cands_lab["group"].map(t)
    sel = cands_lab[cands_lab.conf >= thr][E.CAND_COLS]
    r = E.fppi_recall(sel, gts, 0.0)
    r["objective"] = (r["recall_q1"] + r["recall_overlap"]) / 2
    sc = single_curve.sort_values(["fppi", "recall"])
    r["single_recall_same_fppi"] = float(np.interp(r["fppi"], sc.fppi, sc.recall))
    r["ok"] = (r["fppi"] <= FPPI_MAX) and (r["recall"] >= r["single_recall_same_fppi"] - 1e-12)
    return r



INSAMPLE_NOTE = "참고용(in-sample): B 전체로 고르고 B 전체에서 평가한 값"
CV_NOTE = "B 내부 5-fold 교차검증 (segment 단위, seed 42): 4 fold로 고르고 빠진 fold에서 평가"
N_FOLDS = 5
CV_SEED = 42
METRICS = ["fppi", "recall", "recall_q1", "recall_overlap"]


def subset(c: pd.DataFrame, gts: pd.DataFrame, ids) -> tuple[pd.DataFrame, pd.DataFrame]:
    """image_id 목록으로 후보(라벨 포함)·정답을 자른다. gts 인덱스는 유지(gt_idx 참조용)."""
    ids = set(ids)
    cs = c[c.image_id.isin(ids)]
    gs = gts[gts.image_id.isin(ids)].copy()
    gs.attrs["image_ids"] = sorted(ids)
    return cs, gs


def select(c: pd.DataFrame, gts: pd.DataFrame) -> dict:
    """라벨·그룹이 붙은 후보 c와 정답 gts로 조합 선택 (빠른 탐색 -> 실제 매칭 확인)."""
    n_img = len(gts.attrs["image_ids"])
    n_gt, n_q1, n_ov = len(gts), int(gts.q1.sum()), int(gts.overlap.sum())
    single = E.sweep(c[E.CAND_COLS], gts, SINGLE_THRS)
    ranked, n_ok = fast_search(group_tables(c, gts), n_img, n_gt, n_q1, n_ov, single)
    if ranked.empty:
        raise RuntimeError("FPPI <= 0.5 와 단일 합격선 제약을 동시에 만족하는 조합이 없음")
    tried = []
    for rank, row in ranked.head(MAX_VERIFY).iterrows():
        t = {"small": row.t_small, "normal_ov": row.t_normal_ov, "normal_noov": row.t_normal_noov}
        r = real_eval(c, gts, t, single)
        tried.append({"rank": rank, **t, "fast_objective": row.objective, "fast_fppi": row.fppi,
                      "fast_recall": row.recall, "real_objective": r["objective"], "real_fppi": r["fppi"],
                      "real_recall": r["recall"], "real_ok": r["ok"]})
        if r["ok"]:
            return dict(rank=rank, t=t, row=row, real=r, single=single, ranked=ranked,
                        tried=pd.DataFrame(tried), n_ok=n_ok)
    raise RuntimeError(f"상위 {MAX_VERIFY}개 조합 모두 실제 매칭에서 조건 위반")


def evaluate(c: pd.DataFrame, gts: pd.DataFrame, t: dict) -> dict:
    """정해진 그룹별 합격선 t를 (c, gts)에 적용 + 같은 FPPI의 단일 합격선과 비교."""
    single = E.sweep(c[E.CAND_COLS], gts, SINGLE_THRS)
    sel = c[c.conf >= c.group.map(t)][E.CAND_COLS]
    r = E.fppi_recall(sel, gts, 0.0)
    out = {f"group_{k}": r[k] for k in METRICS}
    out["single_fppi"] = r["fppi"]
    for k in METRICS[1:]:
        out[f"single_{k}"] = E.recall_at_fppi(single, r["fppi"], k)
    return out


def labeled(cands: pd.DataFrame, gts: pd.DataFrame) -> pd.DataFrame:
    c = candidate_labels(cands, gts)
    c["group"] = assign_groups(c)
    return c


def run(cands: pd.DataFrame, gts: pd.DataFrame, tag: str, out_dir: str, model: str = "",
        c: pd.DataFrame | None = None) -> dict:
    """참고용(in-sample): B 전체로 고르고 B 전체에서 평가."""
    os.makedirs(out_dir, exist_ok=True)
    c = labeled(cands, gts) if c is None else c
    s = select(c, gts)
    rank, t, row, r, single = s["rank"], s["t"], s["row"], s["real"], s["single"]
    pre = os.path.join(out_dir, f"E3_insample_")
    E.save_csv(s["tried"], f"{pre}verify_{tag}.csv", model, INSAMPLE_NOTE)
    E.save_csv(s["ranked"].head(20), f"{pre}top20_fast_{tag}.csv", model, INSAMPLE_NOTE)
    s05 = {k: E.recall_at_fppi(single, FPPI_MAX, k) for k in ("recall", "recall_q1", "recall_overlap")}
    summary = pd.DataFrame([
        {"method": "group_thresholds(real)", **{f"t_{g}": t[g] for g in GROUPS},
         "fppi": r["fppi"], "recall": r["recall"], "recall_q1": r["recall_q1"],
         "recall_overlap": r["recall_overlap"], "objective": r["objective"]},
        {"method": "group_thresholds(fast)", **{f"t_{g}": t[g] for g in GROUPS},
         "fppi": row.fppi, "recall": row.recall, "recall_q1": row.recall_q1,
         "recall_overlap": row.recall_overlap, "objective": row.objective},
        {"method": "single_threshold@same_fppi", "fppi": r["fppi"], "recall": r["single_recall_same_fppi"],
         **{k: E.recall_at_fppi(single, r["fppi"], k) for k in ("recall_q1", "recall_overlap")}},
        {"method": "single_threshold@fppi0.5", "fppi": FPPI_MAX, **s05,
         "objective": (s05["recall_q1"] + s05["recall_overlap"]) / 2},
    ])
    summary["n_valid_combos"] = s["n_ok"]
    summary["verify_rank"] = rank
    E.save_csv(summary, f"{pre}result_{tag}.csv", model, INSAMPLE_NOTE)
    group_n = c.group.value_counts().reindex(GROUPS).fillna(0).astype(int).to_dict()
    return dict(summary=summary, n_valid=s["n_ok"], verify_rank=rank, group_n=group_n, ranked=s["ranked"])


def assign_folds(split_b: pd.DataFrame, k: int = N_FOLDS, seed: int = CV_SEED) -> pd.DataFrame:
    """segment 단위로 k fold 배정 (이미지 수 균형).

    segment 순서를 seed로 섞은 뒤 이미지 수 내림차순(동률은 섞인 순서)으로,
    현재 이미지 수가 가장 적은 fold에 하나씩 넣는다 (greedy). 반환: [segment_id, n_images, fold]
    """
    seg = split_b.groupby("segment_id").size().rename("n_images").reset_index()
    rng = np.random.default_rng(seed)
    seg = seg.iloc[rng.permutation(len(seg))].reset_index(drop=True)
    seg = seg.sort_values("n_images", ascending=False, kind="stable").reset_index(drop=True)
    if len(seg) < k:
        raise ValueError(f"segment 수({len(seg)})가 fold 수({k})보다 적음")
    load = np.zeros(k, int)
    folds = []
    for n in seg.n_images:
        f = int(np.argmin(load))                      # 동률이면 번호 작은 fold
        folds.append(f)
        load[f] += n
    seg["fold"] = folds
    return seg


def run_cv(cands: pd.DataFrame, gts: pd.DataFrame, split_b: pd.DataFrame, tag: str, out_dir: str,
           model: str = "", c: pd.DataFrame | None = None) -> dict:
    """B 내부 segment 단위 k-fold 교차검증."""
    os.makedirs(out_dir, exist_ok=True)
    c = labeled(cands, gts) if c is None else c
    split_b = split_b.assign(image_id=split_b.image_id.astype(str).str.zfill(6))
    n_seg = split_b.segment_id.nunique()
    n_folds = min(N_FOLDS, n_seg)                            # segment가 5개 미만이면 fold 수 = segment 수
    if n_folds < 2:
        raise ValueError(f"segment가 {n_seg}개라 교차검증 불가")
    seg = assign_folds(split_b, k=n_folds)
    note = CV_NOTE.replace(f"{N_FOLDS}-fold", f"{n_folds}-fold")
    img_fold = split_b.merge(seg[["segment_id", "fold"]], on="segment_id")[["image_id", "fold"]]
    all_ids = set(gts.attrs["image_ids"])
    if set(img_fold.image_id) != all_ids:
        raise ValueError("split.csv의 B 이미지와 정답 이미지 목록이 다름")

    rows = []
    for f in range(n_folds):
        held = img_fold.loc[img_fold.fold == f, "image_id"]
        train = img_fold.loc[img_fold.fold != f, "image_id"]
        c_tr, g_tr = subset(c, gts, train)
        c_te, g_te = subset(c, gts, held)
        s = select(c_tr, g_tr)
        ev = evaluate(c_te, g_te, s["t"])
        rows.append({"fold": f, "n_images_train": len(train), "n_images_test": len(held),
                     "n_segments_test": int((seg.fold == f).sum()), "n_gt_test": len(g_te),
                     "n_q1_test": int(g_te.q1.sum()), "n_overlap_test": int(g_te.overlap.sum()),
                     **{f"t_{g}": s["t"][g] for g in GROUPS},
                     "train_objective": s["real"]["objective"], "train_fppi": s["real"]["fppi"],
                     "verify_rank": s["rank"], **ev})
    folds = pd.DataFrame(rows)
    for k in METRICS[1:]:
        folds[f"diff_{k}"] = folds[f"group_{k}"] - folds[f"single_{k}"]

    summ = []
    for method in ("group", "single", "diff"):
        ks = METRICS if method != "diff" else METRICS[1:]
        r = {"method": {"group": "group_thresholds", "single": "single_threshold@same_fppi",
                        "diff": "group - single"}[method]}
        for k in ks:
            v = folds[f"{method}_{k}"]
            r[f"{k}_mean"], r[f"{k}_std"] = v.mean(), v.std(ddof=1)
            r[k] = f"{v.mean():.4f} ± {v.std(ddof=1):.4f}"
        summ.append(r)
    summary = pd.DataFrame(summ)

    pre = os.path.join(out_dir, "E3_cv_")
    E.save_csv(folds, f"{pre}folds_{tag}.csv", model, note)
    E.save_csv(summary, f"{pre}summary_{tag}.csv", model, note + " / 평균 ± 표준편차(ddof=1)")
    E.save_csv(seg.sort_values(["fold", "segment_id"]), f"{pre}fold_assignment_{tag}.csv", model, note)
    return dict(folds=folds, summary=summary, seg=seg, img_fold=img_fold, k=n_folds)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cands")
    ap.add_argument("--mode", choices=["insample", "cv", "both"], default="both")
    a = ap.parse_args()
    model, split = E.parse_cand_name(a.cands)
    if split != "B":
        sys.exit(f"E3는 B 후보로만 실행한다 (받은 split={split!r}). Test/proxy 후보는 금지.")
    tag, out = f"{model}_{split}", os.path.join(E.RESULTS_DIR, "E3")
    cands, gts = E.load_candidates(a.cands), E.load_gt("B")
    c = labeled(cands, gts)
    pd.set_option("display.width", 220)
    if a.mode in ("insample", "both"):
        res = run(cands, gts, tag, out, model, c=c)
        print("[참고용(in-sample)] groups:", res["group_n"], "valid:", res["n_valid"], "rank:", res["verify_rank"])
        print(res["summary"].round(4).to_string(index=False))
    if a.mode in ("cv", "both"):
        sp = pd.read_csv(E.SPLIT_CSV, dtype={"image_id": str})
        cv = run_cv(cands, gts, sp[sp.split == "B"], tag, out, model, c=c)
        print("[CV] fold별 합격선·평가")
        print(cv["folds"][["fold", "n_images_test", "t_small", "t_normal_ov", "t_normal_noov",
                           "group_fppi", "group_recall", "group_recall_q1", "group_recall_overlap",
                           "single_recall", "single_recall_q1", "single_recall_overlap"]].round(4).to_string(index=False))
        print(cv["summary"][["method", "fppi", "recall", "recall_q1", "recall_overlap"]].fillna("").to_string(index=False))
