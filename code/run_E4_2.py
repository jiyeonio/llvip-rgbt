"""E4-2: 장면 의존 특징(위치·절대 밝기)을 뺀 재점수화. B로만 실행한다.

가설: E4-1의 LightGBM이 나빴던 이유는 장면별 위치·밝기를 외웠기 때문이다.
특징: ① y중심 제거 + logit(conf), D 그대로, ②③ 상대값만.
모델: 로지스틱 회귀(메인), 제약 LightGBM, 진단(E4-1 LightGBM 설정 + ①에서 y중심만 제거).
평가: E4-1과 같은 함수·fold·FPPI. 장면(segment)별 표 추가.

python code/run_E4_2.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval as E  # noqa: E402
import run_E3 as R  # noqa: E402
import run_E4 as X  # noqa: E402
from run_E2 import candidate_labels  # noqa: E402

EXTRA_CACHE = os.path.join(E.ROOT, "data", f"features_{X.TAG}_e42.parquet")
NOTE = "E4-2 (avg, B) 5-fold: E3와 같은 segment 단위 fold 배정, 빠진 fold에서 평가"

CONFIGS = {
    "box": E.FEAT_BOX_E42,
    "box+dup": E.FEAT_BOX_E42 + E.FEAT_DUP,
    "box+dup+ir": E.FEAT_BOX_E42 + E.FEAT_DUP + E.FEAT_IR_E42,
    "box+dup+ir+rgb": E.FEAT_BOX_E42 + E.FEAT_DUP + E.FEAT_IR_E42 + E.FEAT_RGB_E42,
}
LGBM_C = dict(num_leaves=7, min_child_samples=200, n_estimators=300, learning_rate=0.05,
              random_state=0, n_jobs=8, verbose=-1)
DIAG = "diag LGBM(E4-1 설정) box−yc"
DIAG_COLS = [c for c in E.FEAT_BOX if c != "yc"]


def models():
    """(이름, 특징 열, 생성 함수) 목록."""
    from lightgbm import LGBMClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    out = []
    for cfg, cols in CONFIGS.items():
        out.append((f"LR {cfg}", cols, lambda cols=cols: make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))))
    for cfg, cols in CONFIGS.items():
        mono = [1 if c in ("conf", "logit_conf") else 0 for c in cols]
        out.append((f"LGBM-c {cfg}", cols, lambda mono=mono: LGBMClassifier(monotone_constraints=mono, **LGBM_C)))
    out.append((DIAG, DIAG_COLS, lambda: LGBMClassifier(**X.LGBM_PARAMS)))
    return out


def load_e42(feats: pd.DataFrame) -> pd.DataFrame:
    if os.path.exists(EXTRA_CACHE):
        ex = pd.read_parquet(EXTRA_CACHE)
        if len(ex) == len(feats) and set(ex.cand_idx) == set(feats.cand_idx):
            print(f"E4-2 추가 특징 캐시 사용: {EXTRA_CACHE}")
            return E.build_e42(feats, ex)
    ex = E.extract_e42_extra(feats, "train")
    assert np.allclose(ex.set_index("cand_idx").loc[feats.cand_idx, "rgb_mean_chk"].to_numpy(), feats.rgb_mean.to_numpy())
    ex.drop(columns="rgb_mean_chk").to_parquet(EXTRA_CACHE, index=False)
    print(f"E4-2 추가 특징 {len(ex)}개 -> {EXTRA_CACHE}")
    return E.build_e42(feats, ex.drop(columns="rgb_mean_chk"))


def evaluate_rows(c_new_te, c_all, gts, ids, base_cache, key):
    """ids 이미지에서 기준 곡선과 새 점수 곡선을 그려 FPPI 목표별 Recall."""
    c_base, g = R.subset(c_all, gts, ids)
    if key not in base_cache:
        base_cache[key] = E.sweep(c_base[E.CAND_COLS], g, R.SINGLE_THRS)
    base = base_cache[key]
    cur = X.curve_new(c_new_te[c_new_te.image_id.isin(ids)], g)
    rows = []
    for ft in X.FPPI_TARGETS:
        a, b = X.at_fppi(cur, ft), X.at_fppi(base, ft)
        rows.append({"fppi": ft, "n_images": len(ids), "n_gt": len(g), "n_q1": int(g.q1.sum()),
                     "n_overlap": int(g.overlap.sum()),
                     **{f"new_{k}": a[k] for k in X.GROUPS_KO}, **{f"base_{k}": b[k] for k in X.GROUPS_KO},
                     **{f"diff_{k}": a[k] - b[k] for k in X.GROUPS_KO}})
    return rows


def main():
    os.makedirs(X.OUT, exist_ok=True)
    gts = E.load_gt(X.SPLIT)
    cands = E.load_candidates(X.CAND_PATH)
    c_all = candidate_labels(cands, gts)
    feats = X.load_features(cands)                         # 기존 캐시 재사용
    data = load_e42(feats)
    data = pd.concat([data, c_all.loc[data.cand_idx, ["label"]].reset_index(drop=True)], axis=1)
    fa = pd.read_csv(X.FOLD_FILE, skiprows=1)
    sp = pd.read_csv(E.SPLIT_CSV, dtype={"image_id": str})
    spb = sp[sp.split == X.SPLIT].merge(fa[["segment_id", "fold"]], on="segment_id")
    img_fold = spb.set_index("image_id").fold
    img_seg = spb.set_index("image_id").segment_id
    data["fold"] = data.image_id.map(img_fold).to_numpy()
    assert data.fold.notna().all()
    folds = sorted(data.fold.unique())
    print(f"대상 후보 {len(data)}개 (양성 {int(data.label.sum())}), 결측 비율 최대 "
          f"{data[CONFIGS['box+dup+ir+rgb']].isna().mean().max():.4f}")

    fold_rows, scene_rows, gains, coefs = [], [], [], []
    base_cache = {}
    for f in folds:
        tr, te = data.fold != f, data.fold == f
        held = set(img_fold[img_fold == f].index)
        segs = sorted(img_seg[img_seg.index.isin(held)].unique())
        for name, cols, mk in models():
            m = mk().fit(data.loc[tr, cols], data.loc[tr, "label"])
            s = m.predict_proba(data.loc[te, cols])[:, 1]
            c_new = data.loc[te, ["image_id", "x1", "y1", "x2", "y2"]].assign(conf=s)
            for r in evaluate_rows(c_new, c_all, gts, held, base_cache, ("fold", f)):
                fold_rows.append({"fold": f, "model": name, **r})
            for sg in segs:
                ids = set(img_seg[img_seg == sg].index)
                for r in evaluate_rows(c_new, c_all, gts, ids, base_cache, ("seg", sg)):
                    scene_rows.append({"fold": f, "segment_id": sg, "model": name, **r})
            if name.startswith("LR"):
                lr = m[-1]
                coefs.append(pd.DataFrame({"model": name, "fold": f, "feature": cols, "coef": lr.coef_[0]}))
            else:
                gains.append(pd.DataFrame({"model": name, "fold": f, "feature": cols,
                                           "gain": m.booster_.feature_importance("gain")}))
        print(f"fold {f} 완료 (segment {segs})")

    fold_tab, scene_tab = pd.DataFrame(fold_rows), pd.DataFrame(scene_rows)
    E.save_csv(fold_tab, os.path.join(X.OUT, f"E4_2_folds_{X.TAG}.csv"), X.MODEL, NOTE)
    E.save_csv(scene_tab, os.path.join(X.OUT, f"E4_2_scenes_{X.TAG}.csv"), X.MODEL,
               NOTE + " / 장면(segment)별: 그 segment 이미지만으로 기준·새 점수 곡선")

    summ = []
    for (n, ft), g in fold_tab.groupby(["model", "fppi"], sort=False):
        r = {"model": n, "fppi": ft}
        for k in X.GROUPS_KO:
            for p in ("new", "base", "diff"):
                r[f"{p}_{k}_mean"], r[f"{p}_{k}_std"] = X.mean_std(g[f"{p}_{k}"])
        summ.append(r)
    summary = pd.DataFrame(summ)
    E.save_csv(summary, os.path.join(X.OUT, f"E4_2_summary_{X.TAG}.csv"), X.MODEL, NOTE + " / 평균 ± 표준편차(ddof=1)")

    imp = pd.concat(gains).groupby(["model", "feature"], sort=False).gain.mean().reset_index()
    imp["rank"] = imp.groupby("model").gain.rank(ascending=False, method="first").astype(int)
    E.save_csv(imp[imp["rank"] <= 15].sort_values(["model", "rank"]),
               os.path.join(X.OUT, f"E4_2_importance_{X.TAG}.csv"), X.MODEL, "LightGBM 특징 중요도(gain), 5 fold 평균, 상위 15개")
    co = pd.concat(coefs).groupby(["model", "feature"], sort=False).coef.agg(["mean", "std"]).reset_index()
    co["abs_rank"] = co.groupby("model")["mean"].transform(lambda v: v.abs().rank(ascending=False, method="first")).astype(int)
    E.save_csv(co.sort_values(["model", "abs_rank"]), os.path.join(X.OUT, f"E4_2_lr_coef_{X.TAG}.csv"), X.MODEL,
               "로지스틱 회귀 계수 (표준화 기준), 5 fold 평균·표준편차")

    # E4-1과 나란히 (FPPI 0.5)
    e41 = pd.read_csv(os.path.join(X.OUT, f"E4_1_summary_{X.TAG}.csv"), skiprows=1)
    rows = []

    def add(exp, name, r, prefix):
        d = {"실험": exp, "모델": name}
        for k in X.GROUPS_KO:
            d[k] = f"{r[f'{prefix}_{k}_mean']:.4f} ± {r[f'{prefix}_{k}_std']:.4f}"
            d[f"{k}_diff_%p"] = (f"{100 * r[f'diff_{k}_mean']:+.2f} ± {100 * r[f'diff_{k}_std']:.2f}"
                                 if prefix == "new" else "+0.00 ± 0.00")
            d[f"{k}_diff_mean_%p"] = 100 * r[f"diff_{k}_mean"] if prefix == "new" else 0.0
        rows.append(d)

    s41 = e41[e41.fppi == 0.5]
    add("기준", "단일 합격선 (원래 conf)", s41.iloc[0], "base")
    for _, r in s41.iterrows():
        add("E4-1", ("LGBM " + r.config) if not r.config.startswith("LR") else r.config, r, "new")
    for _, r in summary[summary.fppi == 0.5].iterrows():
        add("E4-2", r.model, r, "new")
    comp = pd.DataFrame(rows)
    E.save_csv(comp, os.path.join(X.OUT, f"E4_2_compare_{X.TAG}.csv"), X.MODEL,
               "E4-1 vs E4-2, FPPI 0.5, 5-fold 평균 ± 표준편차, 차이 = 같은 fold·같은 FPPI 원래 conf 단일 합격선 대비(%p)")
    pd.set_option("display.width", 250)
    print(comp[["실험", "모델", "recall_diff_%p", "recall_q1_diff_%p", "recall_overlap_diff_%p"]].to_string(index=False))


if __name__ == "__main__":
    main()
