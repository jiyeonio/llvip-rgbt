"""해커톤 채점 하니스. code/eval.py 의 평가 함수를 그대로 쓴다 (eval.py 는 수정하지 않음).

고정된 것 (건드리지 않음)
  - 평가 후보: candidates/candidates_avg_B.csv  (B 장면, 탐지기는 A로만 학습)
  - 채점: eval.load_gt("B"), eval.sweep, eval.recall_at_fppi  (FPPI 0.25 / 0.5 / 1.0)
  - Test 후보는 열지 않는다.

각자 바꾸는 것: train(env, seed) -> state,  rescore(state, cands, env) -> 새 점수(np.ndarray)
  - rescore 는 후보 박스는 그대로 두고 점수(conf)만 새로 매긴다. 정답(B)은 볼 수 없다.

사용: hackathon.ipynb 참고. 기준선 값 확인:  .venv/bin/python hackathon/lb.py --baseline
"""
from __future__ import annotations

import json
import os
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("LLVIP_EVAL_ROOT", os.path.dirname(HERE))      # /workspace/llvip_eval
sys.path.insert(0, os.path.join(ROOT, "code"))
sys.path.insert(0, HERE)
import eval as E  # noqa: E402  (code/eval.py)
import board  # noqa: E402

# ---------------------------------------------------------------- 고정 설정
EVAL_SPLIT = "B"
EVAL_CANDS = os.path.join(ROOT, "candidates", "candidates_avg_B.csv")
TRAIN_SPLIT = "A"
TRAIN_CANDS = os.path.join(ROOT, "candidates", "candidates_avg_A.csv")   # predict.py 로 만들어야 함
RESCORE_RANGE = (0.01, 0.5)          # 엑셀 '공통 과정' 4단계: conf 0.01~0.5 후보가 재점수 대상 (B 기준 26,453개)
FPPI_TARGETS = (0.25, 0.5, 1.0)
UPPER_Q1_PP = 15.2                   # 엑셀 '참조-상한' 행: 상한 ΔQ1 = +15.2%p (E4-0)
SHEET_BASELINE = dict(q1=0.685, overlap=0.751, all=0.873)   # 엑셀 '참조-기준선' (E1)
SHEET_UPPER = dict(q1=0.837, overlap=0.904, all=0.946, d_q1=15.2, d_overlap=15.3, d_all=7.3)
SHEET_N_RESCORE_B = 26453
assert "DO_NOT_OPEN" not in EVAL_CANDS and "test" not in os.path.basename(EVAL_CANDS).lower()


# ---------------------------------------------------------------- 곡선·지표
def _curve(scored, gts):
    """새 점수(conf 열)로 합격선을 훑은 곡선 (run_E4.curve_new 와 같은 방식)."""
    thrs = np.unique(np.quantile(scored["conf"].to_numpy(), np.linspace(0, 1, 4001)))
    return E.sweep(scored[E.CAND_COLS], gts, thrs)


def _thr_at(curve, target):
    ok = curve[curve.fppi <= target]
    return float(ok.thr.min()) if len(ok) else float("nan")


def _scenes(gts):
    import pandas as pd
    sp = pd.read_csv(E.SPLIT_CSV, dtype={"image_id": str})
    sp["image_id"] = sp["image_id"].str.zfill(6)
    ids = set(gts.attrs["image_ids"])
    sp = sp[sp.image_id.isin(ids)]
    return {int(s): sorted(g.image_id) for s, g in sp.groupby("segment_id")}


def _scene_q1(scored, gts, thr, scenes):
    out = {}
    for sc, ids in scenes.items():
        c = scored[scored.image_id.isin(ids)].reset_index(drop=True)
        g = gts[gts.image_id.isin(ids)].reset_index(drop=True)
        g.attrs["image_ids"] = ids
        out[sc] = E.fppi_recall(c, g, thr)["recall_q1"] if len(g) else float("nan")
    return out


def _metrics(scored, gts, scenes):
    cur = _curve(scored, gts)
    at = lambda f, col: E.recall_at_fppi(cur, f, col)
    thr = _thr_at(cur, 0.5)
    return dict(q1=at(0.5, "recall_q1"), overlap=at(0.5, "recall_overlap"), all=at(0.5, "recall"),
                all_025=at(0.25, "recall"), all_10=at(1.0, "recall"), thr05=thr,
                scene_q1=_scene_q1(scored, gts, thr, scenes))


# ---------------------------------------------------------------- 기준선
BASELINE_PATH = board.BASE_JSON


def compute_baseline(force=False, verbose=True):
    """원래 conf 그대로의 기준선을 계산해 baseline.json 에 저장 (후보 파일이 같으면 재사용)."""
    mtime = os.path.getmtime(EVAL_CANDS)
    if os.path.exists(BASELINE_PATH) and not force:
        with open(BASELINE_PATH, encoding="utf-8") as f:
            b = json.load(f)
        if b.get("cands_mtime") == mtime:
            return b
    gts = E.load_gt(EVAL_SPLIT)
    cands = E.load_candidates(EVAL_CANDS)
    m = _metrics(cands, gts, _scenes(gts))
    lo, hi = RESCORE_RANGE
    n_range = int(((cands.conf >= lo) & (cands.conf <= hi)).sum())
    b = dict(m, cands_mtime=mtime, n_images=len(gts.attrs["image_ids"]), n_cands=len(cands), n_rescore_range=n_range,
             upper=SHEET_UPPER)
    b["scene_q1"] = {str(k): v for k, v in m["scene_q1"].items()}
    with open(BASELINE_PATH, "w", encoding="utf-8") as f:
        json.dump(b, f, ensure_ascii=False, indent=1)
    if verbose:
        print("기준선(B, FPPI 0.5): Q1 %.3f / 겹침 %.3f / 전체 %.3f   (엑셀 참조-기준선: %.3f / %.3f / %.3f)" % (
            b["q1"], b["overlap"], b["all"], *SHEET_BASELINE.values()))
        bad = [k for k in SHEET_BASELINE if abs(b[k] - SHEET_BASELINE[k]) > 0.01]
        print("  -> 엑셀 값과 0.01 이내로 일치" if not bad else f"  -> 주의: {bad} 가 엑셀 값과 0.01 넘게 다름. 곡선 보간 방식 차이인지 확인 필요")
        print(f"재점수 대상(conf {lo}~{hi}) 후보 수: {n_range:,} (엑셀: {SHEET_N_RESCORE_B:,})")
    return b


# ---------------------------------------------------------------- 환경 (train / rescore 에 넘겨줌)
class Env:
    """train / rescore 에서 쓰는 도우미. 정답은 A 것만 줄 수 있다."""
    E = E
    root = ROOT
    llvip_dir = E.LLVIP_DIR
    rescore_range = RESCORE_RANGE
    split_csv = E.SPLIT_CSV

    def __init__(self):
        try:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            self.device = "cpu"

    def rescore_mask(self, cands):
        lo, hi = RESCORE_RANGE
        return ((cands["conf"] >= lo) & (cands["conf"] <= hi)).to_numpy()

    def train_cands(self):
        """A 장면 후보. 없으면 predict.py 로 먼저 만들어야 한다."""
        if not os.path.exists(TRAIN_CANDS):
            raise FileNotFoundError(
                f"{TRAIN_CANDS} 가 없습니다. weights/A_avg_best.pt 로 A 장면 이미지에 code/predict.py 를 돌려 "
                "같은 형식(image_id,x1,y1,x2,y2,conf; conf>=0.001, 원본 1280x1024 좌표)으로 만든 뒤 다시 실행하세요.")
        c = E.load_candidates(TRAIN_CANDS)
        import pandas as pd
        sp = pd.read_csv(E.SPLIT_CSV, dtype={"image_id": str})
        a_ids = set(sp.loc[sp.split == TRAIN_SPLIT, "image_id"].str.zfill(6))
        extra = set(c.image_id) - a_ids
        if extra:
            raise ValueError(f"A 후보 파일에 A 장면이 아닌 이미지가 {len(extra)}장 있음 (예: {sorted(extra)[:3]}). B/Test 누수 방지를 위해 중단")
        return c

    def train_gts(self):
        return E.load_gt(TRAIN_SPLIT)


# ---------------------------------------------------------------- 유틸
def _seed(s):
    random.seed(s)
    np.random.seed(s)
    try:
        import torch
        torch.manual_seed(s)
        torch.cuda.manual_seed_all(s)
    except Exception:
        pass


def _sync():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def count_params_m(state):
    """state 가 모델이거나 {'model': 모델} 이면 파라미터 수(백만). 아니면 nan."""
    m = state.get("model") if isinstance(state, dict) else state
    try:
        return sum(p.numel() for p in m.parameters()) / 1e6
    except Exception:
        return float("nan")


# ---------------------------------------------------------------- 제출
def submit(id, owner, change, setting, train, rescore, seeds=(0, 1, 2), status="완료", note="", dry_run=False):
    """시드마다 train -> rescore -> 채점하고, 평균을 리더보드에 한 줄로 기록한다.

    Δ·상한 회수율·최악 장면 Δ 는 기준선(원래 conf, 같은 방식의 곡선) 대비.
    같은 ID로 다시 제출하면 순위표에는 가장 최근 것이 보인다 (전체 기록은 board.py --all).
    """
    if not str(id).strip():
        raise ValueError("ID 를 적어 주세요 (예: R0, R5-a)")
    base = compute_baseline()
    gts = E.load_gt(EVAL_SPLIT)
    cands = E.load_candidates(EVAL_CANDS)
    scenes = _scenes(gts)
    n_img = len(gts.attrs["image_ids"])
    orig = cands["conf"].to_numpy()
    env = Env()
    per = []
    for s in seeds:
        _seed(s)
        t0 = time.time()
        state = train(env, s)
        train_min = (time.time() - t0) / 60
        _sync()
        t1 = time.time()
        new = np.asarray(rescore(state, cands.copy(), env), dtype=float).reshape(-1)
        _sync()
        dt = time.time() - t1
        if new.shape[0] != len(cands) or not np.isfinite(new).all():
            raise ValueError(f"rescore 는 후보 {len(cands)}개와 같은 길이의 유한한 점수를 돌려줘야 합니다 (받은 길이 {new.shape[0]}, 비유한값 {int((~np.isfinite(new)).sum())}개)")
        scored = cands.copy()
        scored["conf"] = new
        m = _metrics(scored, gts, scenes)
        n_changed = int((np.abs(new - orig) > 1e-9).sum())
        m.update(ms_per_cand=(dt * 1000 / n_changed) if n_changed else float("nan"), n_changed=n_changed,
                 params_m=count_params_m(state), train_min=train_min,
                 d_scene={k: 100 * (m["scene_q1"][k] - base["scene_q1"][str(k)]) for k in m["scene_q1"]})
        per.append(m)
        print(f"[시드 {s}] Q1 {m['q1']:.3f} / 겹침 {m['overlap']:.3f} / 전체 {m['all']:.3f}  "
              f"(ΔQ1 {100 * (m['q1'] - base['q1']):+.2f}%p)  재점수 {n_changed:,}개  학습 {train_min:.1f}분")

    def _nm(v):
        v = np.asarray(v, float)
        return float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")
    mean = lambda k: _nm([p[k] for p in per])
    q1s = [p["q1"] for p in per]
    scene_keys = sorted(per[0]["d_scene"])
    d_scene_mean = {k: _nm([p["d_scene"][k] for p in per]) for k in scene_keys}
    dq1 = 100 * (mean("q1") - base["q1"])
    row = dict(id=id, owner=owner, change=change, setting=setting,
               q1=mean("q1"), overlap=mean("overlap"), all=mean("all"),
               d_q1=dq1, d_overlap=100 * (mean("overlap") - base["overlap"]), d_all=100 * (mean("all") - base["all"]),
               upper_recovery=dq1 / UPPER_Q1_PP,
               worst_scene_d_q1=(float(np.nanmin(list(d_scene_mean.values()))) if np.isfinite(list(d_scene_mean.values())).any() else float("nan")),
               q1_seed_std=100 * float(np.std(q1s, ddof=1)) if len(q1s) > 1 else float("nan"),
               all_025=mean("all_025"), all_10=mean("all_10"),
               params_m=mean("params_m"), ms_per_cand=mean("ms_per_cand"),
               status=status, note=note, time=time.strftime("%Y-%m-%d %H:%M:%S"), n_seeds=len(seeds))
    sd = "-" if row["q1_seed_std"] != row["q1_seed_std"] else f"{row['q1_seed_std']:.2f}%p"
    print(f"\n결과 (시드 평균): Q1 {row['q1']:.3f} / 겹침 {row['overlap']:.3f} / 전체 {row['all']:.3f} | "
          f"ΔQ1 {row['d_q1']:+.2f}%p | 상한 회수 {100 * row['upper_recovery']:.0f}% | "
          f"최악 장면 Δ {row['worst_scene_d_q1']:+.2f}%p | σ {sd}")
    fl = board.flag(dict(row, _ref=False))
    if fl == "시드1":
        print("  ※ 시드가 1개라 표준편차를 구할 수 없음: 순위표에서는 참고용으로만 봄 (시드 3개로 다시 제출 권장)")
    elif fl:
        print(f"  ※ {fl}: |ΔQ1| 이 max(1%p, 시드 표준편차) 보다 작아 효과로 보지 않음")
    if dry_run:
        print("DRY_RUN: 리더보드에는 기록하지 않았습니다.")
    else:
        board.append_row(row)
        print("리더보드에 기록했습니다.")
    runs = os.path.join(HERE, "runs")
    os.makedirs(runs, exist_ok=True)
    with open(os.path.join(runs, f"{id}_{time.strftime('%Y%m%d_%H%M%S')}.json"), "w", encoding="utf-8") as f:
        json.dump(dict(row=row, per_seed=[{k: v for k, v in p.items() if k != "scene_q1"} for p in per],
                       d_scene_mean=d_scene_mean, dry_run=dry_run), f, ensure_ascii=False, indent=1, default=float)
    return row


if __name__ == "__main__":
    if "--baseline" in sys.argv:
        compute_baseline(force=True)
    else:
        print(__doc__)