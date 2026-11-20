"""run_E3 가짜 데이터 테스트 (실제 후보 파일은 읽지 않음)."""
import os, subprocess, sys, tempfile
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval as E, run_E3 as R
from run_E2 import candidate_labels

rng = np.random.default_rng(0)
gt_rows, c_rows, ids = [], [], [f"{i:06d}" for i in range(400)]
for iid in ids:
    for _ in range(rng.integers(0, 5)):
        h = rng.choice([rng.uniform(60, 180), rng.uniform(190, 400)])
        w = h * 0.4; x, y = rng.uniform(0, 1280 - w), rng.uniform(0, 1024 - h)
        boxes = [(x, y, x + w, y + h)]
        if rng.random() < 0.2:                       # 겹친 사람
            boxes.append((x + w * 0.4, y, x + w * 1.4, y + h))
        for bx in boxes:
            gt_rows.append((iid, *bx))
            small = (bx[3] - bx[1]) < 184
            conf = np.clip(rng.beta(2, 2) * (0.6 if small else 1.0) + (0.1 if not small else 0), 0.001, 0.99)
            if rng.random() < 0.95:
                j = rng.normal(0, 3, 4); c_rows.append((iid, *(np.array(bx) + j), conf))
            if rng.random() < 0.3:                   # 중복 박스
                c_rows.append((iid, *(np.array(bx) + rng.normal(0, 15, 4)), conf * rng.uniform(0.2, 0.9)))
    for _ in range(rng.integers(0, 8)):              # FP
        h = rng.uniform(50, 300); w = h * 0.4; x, y = rng.uniform(0, 1200), rng.uniform(0, 900)
        c_rows.append((iid, x, y, x + w, y + h, rng.beta(1, 6)))
gts = E.add_gt_flags(pd.DataFrame(gt_rows, columns=["image_id", "x1", "y1", "x2", "y2"]), ids)
cands = pd.DataFrame(c_rows, columns=E.CAND_COLS)
cands["conf"] = cands.conf.clip(0.001, 0.999)
print(f"fake: images={len(ids)} gt={len(gts)} q1={gts.q1.sum()} ov={gts.overlap.sum()} cands={len(cands)}")

# 1) 빠른 누적표 == 라벨로 직접 센 값 (무작위 조합)
c = candidate_labels(cands, gts); c["group"] = R.assign_groups(c)
tab = R.group_tables(c, gts)
gq1 = gts.q1.to_numpy()[gts.index.get_indexer(c.gt_idx.clip(lower=0))]
for _ in range(200):
    ti = rng.integers(0, 60, 3); t = dict(zip(R.GROUPS, R.THRS[ti]))
    keep = c.conf >= c.group.map(t)
    assert sum(tab[g]["tp"][i] for g, i in zip(R.GROUPS, ti)) == (keep & (c.label == 1)).sum()
    assert sum(tab[g]["fp"][i] for g, i in zip(R.GROUPS, ti)) == (keep & (c.label == 0)).sum()
    assert sum(tab[g]["tp_q1"][i] for g, i in zip(R.GROUPS, ti)) == (keep & (c.label == 1) & gq1).sum()
print("1) 누적표 OK")

# 2) 세 합격선이 같으면 (t,t,t) == 단일 합격선 직접 매칭
single = E.sweep(cands, gts, R.SINGLE_THRS)
ranked_all, _ = R.fast_search(tab, len(ids), len(gts), int(gts.q1.sum()), int(gts.overlap.sum()),
                              single.assign(recall=0.0))   # 제약 끄고 전체 조합 확인
for t in (0.05, 0.25, 0.5):
    row = ranked_all[(ranked_all.t_small == t) & (ranked_all.t_normal_ov == t) & (ranked_all.t_normal_noov == t)]
    if row.empty: continue                                   # FPPI > 0.5
    d = E.fppi_recall(cands, gts, t)
    assert abs(row.recall.iloc[0] - d["recall"]) < 1e-12 and abs(row.fppi.iloc[0] - d["fppi"]) < 1e-12
    assert abs(row.recall_q1.iloc[0] - d["recall_q1"]) < 1e-12
print("2) 단일 합격선 일치 OK")

# 3) 전체 실행 (scratchpad 출력) + 결과 검사
out = tempfile.mkdtemp(dir=os.environ.get("SCRATCH", None))
res = R.run(cands, gts, "fake_B", out, "fake")
s = res["summary"].set_index("method")
real = s.loc["group_thresholds(real)"]
assert real.fppi <= R.FPPI_MAX
assert real.recall >= s.loc["single_threshold@same_fppi"].recall - 1e-12
rk = res["ranked"]
assert (rk.fppi <= 0.5).all() and (rk.recall >= rk.single_recall_same_fppi - 1e-12).all()
top = rk.iloc[0]   # 정렬(목표 -> 전체 Recall -> 합격선 합) 확인
assert not ((rk.objective > top.objective) | ((rk.objective == top.objective) & (rk.recall > top.recall))).any()
# 실제 매칭 재계산을 독립적으로 한 번 더
t = {g: real[f"t_{g}"] for g in R.GROUPS}
sel = c[c.conf >= c.group.map(t)][E.CAND_COLS]
d = E.fppi_recall(sel, gts, 0.0)
assert abs(d["recall_q1"] - real.recall_q1) < 1e-12 and abs(d["fppi"] - real.fppi) < 1e-12
print(f"3) 전체 실행 OK  groups={res['group_n']} valid={res['n_valid']} rank={res['verify_rank']}")
print(res["summary"].round(4).to_string(index=False))
files = sorted(os.listdir(out))
print("files:", files)
assert files == ["E3_insample_result_fake_B.csv", "E3_insample_top20_fast_fake_B.csv", "E3_insample_verify_fake_B.csv"]
for f in files:
    assert open(os.path.join(out, f), encoding="utf-8").readline().startswith("참고용(in-sample)"), f

# 4) B가 아닌 후보는 거부 (파일을 읽기 전에 종료)
code = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_E3.py")
p = subprocess.run([sys.executable, code, "candidates/candidates_proxy_test.csv"], capture_output=True, text=True)
assert p.returncode != 0 and "B 후보로만" in p.stderr, p
print("4) Test 후보 거부 OK")

# 5) 교차검증: segment 단위 fold 배정
seg_rows, sid, k = [], 0, 0
while k < len(ids):                                   # 길이 3~40의 연속 구간 = segment
    n = int(rng.integers(3, 41))
    seg_rows += [(i, sid) for i in ids[k:k + n]]; sid += 1; k += n
split_b = pd.DataFrame(seg_rows, columns=["image_id", "segment_id"]).assign(split="B")
seg = R.assign_folds(split_b)
assert seg.segment_id.is_unique and set(seg.fold) == set(range(R.N_FOLDS))
img_fold = split_b.merge(seg[["segment_id", "fold"]], on="segment_id")
assert (img_fold.groupby("segment_id").fold.nunique() == 1).all()          # segment가 fold 사이에 안 섞임
assert len(img_fold) == len(split_b) and img_fold.image_id.is_unique         # 모든 이미지가 정확히 한 fold
load = img_fold.groupby("fold").size()
assert load.max() - load.min() <= seg.n_images.max(), load                   # 이미지 수 균형
assert seg.equals(R.assign_folds(split_b))                                   # seed 42 재현
mp = lambda d: d.set_index("segment_id").fold.sort_index()
assert not mp(seg).equals(mp(R.assign_folds(split_b, seed=7)))              # seed가 실제로 쓰임
print(f"5) fold 배정 OK  segments={len(seg)} images/fold={load.tolist()}")

# 6) 교차검증 실행: 학습/평가 분리, 표, 첫 줄 표시
out_cv = tempfile.mkdtemp(dir=os.environ.get("SCRATCH", None))
cv = R.run_cv(cands, gts, split_b, "fake_B", out_cv, "fake")
fo = cv["folds"]
assert fo.n_images_test.sum() == len(ids) and (fo.n_images_train + fo.n_images_test == len(ids)).all()
assert fo.n_gt_test.sum() == len(gts)
for _, r_ in fo.iterrows():                                                 # 고른 합격선은 격자 위
    assert all(round(r_[f"t_{g}"], 2) in set(R.THRS) for g in R.GROUPS)
sm = cv["summary"].set_index("method")
for kk in R.METRICS:
    assert abs(sm.loc["group_thresholds", f"{kk}_mean"] - fo[f"group_{kk}"].mean()) < 1e-12
    assert abs(sm.loc["group_thresholds", f"{kk}_std"] - fo[f"group_{kk}"].std(ddof=1)) < 1e-12
assert np.allclose(fo.single_fppi, fo.group_fppi)                           # 같은 FPPI 비교
# held-out 평가를 독립 계산과 대조
c_all = R.labeled(cands, gts)
for _, r_ in fo.iterrows():
    hid = set(cv["img_fold"].loc[cv["img_fold"].fold == r_.fold, "image_id"])
    cs, gs = R.subset(c_all, gts, hid)
    t = {g: r_[f"t_{g}"] for g in R.GROUPS}
    d = E.fppi_recall(cs[cs.conf >= cs.group.map(t)][E.CAND_COLS], gs, 0.0)
    assert abs(d["recall_q1"] - r_.group_recall_q1) < 1e-12 and abs(d["fppi"] - r_.group_fppi) < 1e-12
files = sorted(os.listdir(out_cv))
assert files == ["E3_cv_fold_assignment_fake_B.csv", "E3_cv_folds_fake_B.csv", "E3_cv_summary_fake_B.csv"], files
for f in files:
    assert open(os.path.join(out_cv, f), encoding="utf-8").readline().startswith("B 내부 5-fold"), f
print("6) 교차검증 실행 OK")
print(fo[["fold", "n_images_test", "t_small", "t_normal_ov", "t_normal_noov", "group_fppi",
          "group_recall_q1", "single_recall_q1"]].round(3).to_string(index=False))
print(cv["summary"][["method", "fppi", "recall", "recall_q1", "recall_overlap"]].fillna("").to_string(index=False))

# 7) 누출 없음: 빠진 fold의 후보를 바꿔도 그 fold에서 고른 합격선은 그대로
f0 = set(cv["img_fold"].loc[cv["img_fold"].fold == 0, "image_id"])
cands2 = cands.copy()
m0 = cands2.image_id.isin(f0)
cands2.loc[m0, "conf"] = rng.uniform(0.001, 0.999, m0.sum())
cv2 = R.run_cv(cands2, gts, split_b, "fake2_B", tempfile.mkdtemp(dir=os.environ.get("SCRATCH", None)), "fake")
tcols = [f"t_{g}" for g in R.GROUPS]
assert cv2["folds"].loc[0, tcols].equals(fo.loc[0, tcols])
assert not cv2["folds"].loc[0, ["group_recall"]].equals(fo.loc[0, ["group_recall"]])   # 평가는 바뀜
print("7) held-out 누출 없음 OK")

# 8) segment가 5개 미만이면 fold 수 = segment 수
few = split_b.assign(segment_id=split_b.segment_id % 3)
cv3 = R.run_cv(cands, gts, few, "fake3_B", tempfile.mkdtemp(dir=os.environ.get("SCRATCH", None)), "fake")
assert cv3["k"] == 3 and len(cv3["folds"]) == 3 and set(cv3["seg"].fold) == {0, 1, 2}
print("8) segment 3개 -> 3-fold OK")
print("all E3 tests ok")
