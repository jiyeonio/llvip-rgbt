"""E4: 개선 여지 상한(E4-0) + LightGBM 재점수화(E4-1). B로만 실행한다.

E4-0: 원래 conf 단일 합격선의 FPPI 0.5 지점(0.001 간격, FPPI <= 0.5인 가장 낮은 합격선)에서
      놓친 정답 중 IoU >= 0.5 후보가 있는 정답을 모두 살렸다고 가정한 Recall 상한.
E4-1: conf >= 0.01 후보에 특징(① 박스 / D 중복 / ② 열화상 / ③ RGB)을 뽑아 E3와 같은 5-fold로
      재점수화. 새 점수 순서로 1:1 매칭하며 합격선을 훑어 FPPI 0.25/0.5/1.0의 Recall을
      같은 fold·같은 FPPI의 원래 conf 단일 합격선과 비교. 가장 좋은 구성에 E3 그룹별 합격선을 한 번 더 적용.

python code/run_E4.py
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval as E  # noqa: E402
import run_E3 as R  # noqa: E402
from run_E2 import candidate_labels  # noqa: E402

MODEL, SPLIT = "avg", "B"                         # B 후보만. Test 후보 경로는 만들지도 않는다
TAG = f"{MODEL}_{SPLIT}"
CAND_PATH = os.path.join(E.ROOT, "candidates", f"candidates_{TAG}.csv")
FEAT_CACHE = os.path.join(E.ROOT, "data", f"features_{TAG}.parquet")
FOLD_FILE = os.path.join(E.RESULTS_DIR, "E3", f"E3_cv_fold_assignment_{TAG}.csv")
OUT = os.path.join(E.RESULTS_DIR, "E4")
MIN_CONF = 0.01
FPPI_TARGETS = (0.25, 0.5, 1.0)
GROUPS_KO = ("recall", "recall_q1", "recall_overlap")
NOTE = "E4 (avg, B) 5-fold: E3와 같은 segment 단위 fold 배정, 빠진 fold에서 평가"

CONFIGS = {
    "box": E.FEAT_BOX,
    "box+dup": E.FEAT_BOX + E.FEAT_DUP,
    "box+dup+ir": E.FEAT_BOX + E.FEAT_DUP + E.FEAT_IR,
    "box+dup+ir+rgb": E.FEAT_BOX + E.FEAT_DUP + E.FEAT_IR + E.FEAT_RGB,
}
LR_NAME = "LR(box+dup+ir+rgb)"
LGBM_PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=0, verbose=-1)


def make_model(name):
    if name == LR_NAME:
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000))
    from lightgbm import LGBMClassifier
    return LGBMClassifier(**LGBM_PARAMS)


def feature_cols(name):
    return CONFIGS["box+dup+ir+rgb"] if name == LR_NAME else CONFIGS[name]


# ------------------------------------------------------------------ 데이터
def load_features(cands: pd.DataFrame) -> pd.DataFrame:
    """특징 캐시를 읽고 후보 파일과 맞는지 확인. 없거나 다르면 새로 계산해 저장."""
    tgt = cands[cands.conf >= MIN_CONF]
    if os.path.exists(FEAT_CACHE):
        f = pd.read_parquet(FEAT_CACHE)
        same = (len(f) == len(tgt) and (f.cand_idx.to_numpy() == tgt.index.to_numpy()).all()
                and np.allclose(f.conf, tgt.conf) and np.allclose(f.x1, tgt.x1) and np.allclose(f.y2, tgt.y2))
        if same:
            print(f"특징 캐시 사용: {FEAT_CACHE} ({len(f)}개)")
            return f
        print("특징 캐시가 후보 파일과 다름 -> 다시 계산")
    t0 = time.time()
    f, tdf = E.extract_features(cands, MIN_CONF, "train")
    f = f.set_index("cand_idx", drop=False).loc[tgt.index].reset_index(drop=True)
    f.to_parquet(FEAT_CACHE, index=False)
    os.makedirs(OUT, exist_ok=True)
    tsum = pd.DataFrame([{"part": k, "ms_per_image_mean": tdf[c].mean(), "ms_per_image_median": tdf[c].median(),
                          "ms_per_image_p95": tdf[c].quantile(0.95)}
                         for k, c in [("① 박스 + D 중복", "ms_box_dup"), ("② 열화상 (이미지 읽기 포함)", "ms_ir"),
                                      ("③ RGB (이미지 읽기 포함)", "ms_rgb")]])
    tot = tdf.ms_box_dup + tdf.ms_ir + tdf.ms_rgb
    tsum.loc[len(tsum)] = ["합계", tot.mean(), tot.median(), tot.quantile(0.95)]
    tsum["n_images"] = len(tdf)
    tsum["cands_per_image_mean"] = tdf.n_cands.mean()
    E.save_csv(tsum, os.path.join(OUT, f"E4_feature_time_{TAG}.csv"), MODEL,
               "이미지당 특징 추출 시간(ms), CPU 1코어 순차, conf >= 0.01 후보")
    print(f"특징 추출 {len(f)}개, {time.time() - t0:.0f}s -> {FEAT_CACHE}")
    return f


# ------------------------------------------------------------------ 곡선
def curve_new(c_scored: pd.DataFrame, gts: pd.DataFrame) -> pd.DataFrame:
    """새 점수(conf 열)로 1:1 매칭하며 훑은 곡선."""
    s = c_scored.conf.to_numpy()
    thrs = np.unique(np.quantile(s, np.linspace(0, 1, 4001)))
    return E.sweep(c_scored[E.CAND_COLS], gts, thrs)


def at_fppi(curve: pd.DataFrame, f: float) -> dict:
    return {k: E.recall_at_fppi(curve, f, k) for k in GROUPS_KO}


# ------------------------------------------------------------------ E4-0
def upper_bound(cands, gts):
    single = E.sweep(cands, gts, R.SINGLE_THRS)
    ok = single[single.fppi <= 0.5]
    thr = float(ok.thr.min())
    row = ok.loc[ok.thr.idxmin()]
    hit = E.match(cands, gts, thr)["gt_matched"].to_numpy()
    any_all, any_01 = np.zeros(len(gts), bool), np.zeros(len(gts), bool)
    cg = cands.groupby("image_id").indices
    cb = cands[["x1", "y1", "x2", "y2"]].to_numpy(float)
    cc = cands.conf.to_numpy()
    for iid, gpos in gts.groupby("image_id").indices.items():
        p = cg.get(iid)
        if p is None:
            continue
        ious = E.iou_matrix(gts.iloc[gpos][["x1", "y1", "x2", "y2"]].to_numpy(float), cb[p]) >= E.IOU_THR
        any_all[gpos] = ious.any(1)
        any_01[gpos] = (ious & (cc[p] >= MIN_CONF)[None, :]).any(1)
    rows = []
    for name, m in [("전체", np.ones(len(gts), bool)), ("Q1", gts.q1.to_numpy()), ("겹침", gts.overlap.to_numpy())]:
        r_act, r_all, r_01 = hit[m].mean(), (hit | any_all)[m].mean(), (hit | any_01)[m].mean()
        rows.append({"group": name, "n_gt": int(m.sum()), "thr": thr, "fppi": row.fppi, "recall": r_act,
                     "upper_all_cands": r_all, "gain_all_%p": 100 * (r_all - r_act),
                     "upper_conf>=0.01": r_01, "gain_conf>=0.01_%p": 100 * (r_01 - r_act),
                     "n_missed": int((~hit & m).sum()), "n_recoverable": int((~hit & any_all & m).sum())})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 메인
def mean_std(v):
    v = pd.Series(v, dtype=float)
    return v.mean(), v.std(ddof=1)


def main():
    os.makedirs(OUT, exist_ok=True)
    gts = E.load_gt(SPLIT)
    cands = E.load_candidates(CAND_PATH)
    c_all = candidate_labels(cands, gts)                     # E2 라벨 (conf >= 0.001 전체 매칭)
    c_all["group"] = R.assign_groups(c_all)

    # E4-0
    ub = upper_bound(cands, gts)
    E.save_csv(ub, os.path.join(OUT, f"E4_0_upper_bound_{TAG}.csv"), MODEL,
               "E4-0 개선 여지 상한: 원래 conf 단일 합격선 FPPI 0.5 지점에서 IoU>=0.5 후보가 있는 미탐을 모두 살렸다고 가정")
    print(ub.round(4).to_string(index=False))

    # 특징 + 라벨 + fold
    feats = load_features(cands)
    lab = c_all.loc[feats.cand_idx, ["label", "gt_idx", "group"]].reset_index(drop=True)
    data = pd.concat([feats, lab], axis=1)
    fa = pd.read_csv(FOLD_FILE, skiprows=1)
    sp = pd.read_csv(E.SPLIT_CSV, dtype={"image_id": str})
    img_fold = sp[sp.split == SPLIT].merge(fa[["segment_id", "fold"]], on="segment_id").set_index("image_id").fold
    data["fold"] = data.image_id.map(img_fold).to_numpy()
    c_all["fold"] = c_all.image_id.map(img_fold).to_numpy()
    assert data.fold.notna().all() and c_all.fold.notna().all()
    folds = sorted(data.fold.unique())
    print(f"대상 후보 {len(data)}개 (양성 {int(data.label.sum())}), fold {folds}")

    names = list(CONFIGS) + [LR_NAME]
    scores = {n: np.full(len(data), np.nan) for n in names}
    gains, rows = [], []
    for f in folds:
        tr, te = data.fold != f, data.fold == f
        held = set(img_fold[img_fold == f].index)
        c_te_all, g_te = R.subset(c_all, gts, held)
        base = E.sweep(c_te_all[E.CAND_COLS], g_te, R.SINGLE_THRS)
        for n in names:
            cols = feature_cols(n)
            m = make_model(n).fit(data.loc[tr, cols], data.loc[tr, "label"])
            scores[n][te.to_numpy()] = m.predict_proba(data.loc[te, cols])[:, 1]
            if n != LR_NAME:
                gains.append(pd.DataFrame({"config": n, "fold": f, "feature": cols,
                                           "gain": m.booster_.feature_importance("gain")}))
            c_new = data.loc[te, ["image_id", "x1", "y1", "x2", "y2"]].assign(conf=scores[n][te.to_numpy()])
            cur = curve_new(c_new, g_te)
            for ft in FPPI_TARGETS:
                a, b = at_fppi(cur, ft), at_fppi(base, ft)
                rows.append({"fold": f, "config": n, "fppi": ft, "n_images": len(held), "n_gt": len(g_te),
                             **{f"new_{k}": a[k] for k in GROUPS_KO}, **{f"base_{k}": b[k] for k in GROUPS_KO},
                             **{f"diff_{k}": a[k] - b[k] for k in GROUPS_KO}})
        print(f"fold {f} 완료")
    fold_tab = pd.DataFrame(rows)
    E.save_csv(fold_tab, os.path.join(OUT, f"E4_1_folds_{TAG}.csv"), MODEL, NOTE)

    summ = []
    for (n, ft), g in fold_tab.groupby(["config", "fppi"], sort=False):
        r = {"config": n, "fppi": ft}
        for k in GROUPS_KO:
            for p in ("new", "base", "diff"):
                mu, sd = mean_std(g[f"{p}_{k}"])
                r[f"{p}_{k}_mean"], r[f"{p}_{k}_std"] = mu, sd
        r["objective_new_mean"] = ((g.new_recall_q1 + g.new_recall_overlap) / 2).mean()
        summ.append(r)
    summary = pd.DataFrame(summ)
    E.save_csv(summary, os.path.join(OUT, f"E4_1_summary_{TAG}.csv"), MODEL, NOTE + " / 평균 ± 표준편차(ddof=1)")

    imp = pd.concat(gains).groupby(["config", "feature"], sort=False).gain.mean().reset_index()
    imp["rank"] = imp.groupby("config").gain.rank(ascending=False, method="first").astype(int)
    imp = imp[imp["rank"] <= 15].sort_values(["config", "rank"])
    E.save_csv(imp, os.path.join(OUT, f"E4_1_importance_{TAG}.csv"), MODEL, "특징 중요도(gain), 5 fold 평균, 구성별 상위 15개")

    # 가장 좋은 구성 (LightGBM 4개 중, FPPI 0.5 평균 (Q1+겹침)/2 -> 평균 전체 Recall)
    s05 = summary[(summary.fppi == 0.5) & summary.config.isin(CONFIGS)]
    best = s05.sort_values(["objective_new_mean", "new_recall_mean"], ascending=False).config.iloc[0]
    print("가장 좋은 구성:", best)

    # E4+E3: 학습 fold 안쪽 out-of-fold 점수로 합격선 선택 -> 빠진 fold(바깥 모델 점수)에 적용
    cols = feature_cols(best)
    e43 = []
    for f in folds:
        tr_folds = [g for g in folds if g != f]
        inner = np.full(len(data), np.nan)
        for g in tr_folds:
            fit_m = data.fold.isin([h for h in tr_folds if h != g])
            pm = data.fold == g
            inner[pm.to_numpy()] = make_model(best).fit(data.loc[fit_m, cols], data.loc[fit_m, "label"]) \
                .predict_proba(data.loc[pm, cols])[:, 1]
        tr = data.fold != f
        c_tr = data.loc[tr, ["image_id", "x1", "y1", "x2", "y2", "label", "gt_idx", "group"]].assign(conf=inner[tr.to_numpy()])
        tr_ids = set(img_fold[img_fold != f].index)
        _, g_tr = R.subset(c_all, gts, tr_ids)
        held = set(img_fold[img_fold == f].index)
        c_te_all, g_te = R.subset(c_all, gts, held)
        base = E.sweep(c_te_all[E.CAND_COLS], g_te, R.SINGLE_THRS)
        te = data.fold == f
        c_te = data.loc[te, ["image_id", "x1", "y1", "x2", "y2", "label", "gt_idx", "group"]] \
            .assign(conf=scores[best][te.to_numpy()])
        new_cur = curve_new(c_te, g_te)
        try:
            s = R.select(c_tr, g_tr)
        except RuntimeError as ex:
            e43.append({"fold": f, "error": str(ex)})
            continue
        t = s["t"]
        sel = c_te[c_te.conf >= c_te.group.map(t)][E.CAND_COLS]
        r = E.fppi_recall(sel, g_te, 0.0)
        b, nb = at_fppi(base, r["fppi"]), at_fppi(new_cur, r["fppi"])
        e43.append({"fold": f, **{f"t_{g}": t[g] for g in R.GROUPS}, "train_fppi": s["real"]["fppi"],
                    "fppi": r["fppi"], **{k: r[k] for k in GROUPS_KO},
                    **{f"base_{k}": b[k] for k in GROUPS_KO}, **{f"diff_{k}": r[k] - b[k] for k in GROUPS_KO},
                    **{f"best_single_{k}": nb[k] for k in GROUPS_KO}})
    e43 = pd.DataFrame(e43)
    E.save_csv(e43, os.path.join(OUT, f"E4_E3_folds_{TAG}.csv"), MODEL,
               f"E4+E3: {best} 새 점수에 E3 그룹별 합격선 (안쪽 out-of-fold 점수로 선택), 기준 = 같은 FPPI 원래 conf 단일 합격선")

    # 비교 요약표 (FPPI 0.5; E3·E4+E3는 빠진 fold의 실제 FPPI)
    def row(name, fppi, rec, diff):
        r = {"method": name, "fppi": fppi}
        for k in GROUPS_KO:
            mu, sd = mean_std(rec[k])
            dm, dsd = mean_std(diff[k])
            r[k] = f"{mu:.4f} ± {sd:.4f}"
            r[f"{k}_diff_%p"] = f"{100 * dm:+.2f} ± {100 * dsd:.2f}"
            r[f"{k}_mean"], r[f"{k}_diff_mean_%p"] = mu, 100 * dm
        return r

    comp = []
    f05 = fold_tab[fold_tab.fppi == 0.5]
    g0 = f05[f05.config == names[0]]
    comp.append(row("단일 합격선 (원래 conf)", "0.5", {k: g0[f"base_{k}"] for k in GROUPS_KO},
                    {k: g0[f"base_{k}"] * 0 for k in GROUPS_KO}))
    e3 = pd.read_csv(os.path.join(E.RESULTS_DIR, "E3", f"E3_cv_folds_{TAG}.csv"), skiprows=1)
    comp.append(row("E3 그룹별 합격선", f"{e3.group_fppi.mean():.3f} ± {e3.group_fppi.std(ddof=1):.3f}",
                    {k: e3[f"group_{k}"] for k in GROUPS_KO}, {k: e3[f"diff_{k}"] if f"diff_{k}" in e3
                                                              else e3[f"group_{k}"] - e3[f"single_{k}"]
                                                              for k in GROUPS_KO}))
    for n in names:
        g = f05[f05.config == n]
        comp.append(row(f"E4 {n}", "0.5", {k: g[f"new_{k}"] for k in GROUPS_KO},
                        {k: g[f"diff_{k}"] for k in GROUPS_KO}))
    if "fppi" in e43 and e43.fppi.notna().any():
        ok = e43.dropna(subset=["fppi"])
        comp.append(row(f"E4+E3 ({best})", f"{ok.fppi.mean():.3f} ± {ok.fppi.std(ddof=1):.3f}",
                        {k: ok[k] for k in GROUPS_KO}, {k: ok[f"diff_{k}"] for k in GROUPS_KO}))
    comp = pd.DataFrame(comp)
    E.save_csv(comp, os.path.join(OUT, f"E4_compare_{TAG}.csv"), MODEL,
               "비교 요약: 5-fold 평균 ± 표준편차, 차이 = 같은 fold·같은 FPPI 원래 conf 단일 합격선 대비(%p)")
    with open(os.path.join(OUT, f"E4_meta_{TAG}.json"), "w") as fp:
        json.dump({"best_config": best, "lgbm_params": LGBM_PARAMS, "min_conf": MIN_CONF,
                   "n_target": int(len(data)), "folds": [int(x) for x in folds]}, fp, ensure_ascii=False, indent=1)

    pd.set_option("display.width", 250)
    print(comp[["method", "fppi", "recall", "recall_diff_%p", "recall_q1", "recall_q1_diff_%p",
                "recall_overlap", "recall_overlap_diff_%p"]].to_string(index=False))
    print(e43.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
