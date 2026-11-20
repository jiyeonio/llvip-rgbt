"""A로 탐지기 학습 → B·Test 후보 저장까지 한 번에.

설정은 proxy 재학습(train_proxy.py)과 같다:
  입력 = 평균 융합 이미지(RGB:열화상 = 0.5:0.5), yolo11n.pt(COCO), imgsz 640, epochs 28, seed 0,
  deterministic, batch 16, 라벨 = XML -> YOLO person 1클래스
다른 점:
  학습 이미지 = data/split.csv 의 A
  val(best.pt 선택용) = A 안에서 segment 단위로 뗀다 (seed 0으로 segment를 섞고, A 이미지의 10% 이상이 될 때까지).
  B·test는 학습에도 val에도 쓰지 않는다.

출력:
  data/fused_avg/{A,B,test}/                 평균 융합 이미지
  data/yolo_A/                               YOLO 데이터셋 (images=symlink, labels, data.yaml)
  models/<name>/weights/best.pt              학습 결과
  models/<name>/A_val_segments.csv           val로 뗀 segment 기록
  candidates/candidates_<model>_B.csv        B 후보
  candidates/DO_NOT_OPEN_candidates_<model>_test.csv   Test 후보 (열어 보지 말 것, 커밋 금지)

사용:
  python code/train_A.py --plan            # split.csv 검사 + val segment 계획만 출력 (아무것도 쓰지 않음)
  python code/train_A.py                   # 융합 -> 학습 -> B·Test 후보
  python code/train_A.py --skip-train --weights models/A_avg/weights/best.pt   # 후보만 다시 저장
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval import ROOT, llvip_ids  # noqa: E402

SPLITS = ("A", "B", "test")
TRAIN_CFG = dict(imgsz=640, epochs=28, seed=0, deterministic=True, batch=16, workers=8, device=0)


def read_split(path: str) -> pd.DataFrame:
    """split.csv를 읽고 형식을 검사한다. 문제가 있으면 ValueError."""
    sp = pd.read_csv(path, dtype={"image_id": str, "split": str})
    need = {"image_id", "segment_id", "split"}
    if not need <= set(sp.columns):
        raise ValueError(f"split.csv 열 누락: {sorted(need - set(sp.columns))}")
    errs = []
    bad = sorted(set(sp.split) - set(SPLITS))
    if bad:
        errs.append(f"split 값은 A / B / test 만 허용 (받은 다른 값: {bad})")
    if not sp.image_id.str.fullmatch(r"\d{6}").all():
        errs.append(f"6자리가 아닌 image_id {int((~sp.image_id.str.fullmatch(r'[0-9]{6}')).sum())}개 "
                    "(앞의 0이 빠졌는지 확인)")
    if not sp.image_id.is_unique:
        errs.append(f"중복 image_id {int(sp.image_id.duplicated().sum())}개")
    mixed = sp.groupby("segment_id").split.nunique()
    if (mixed > 1).any():
        errs.append(f"여러 split에 걸친 segment: {mixed[mixed > 1].index.tolist()[:10]}")
    train_ids, test_ids = set(llvip_ids("train")), set(llvip_ids("test"))
    for s, pool, name in (("A", train_ids, "train"), ("B", train_ids, "train"), ("test", test_ids, "test")):
        ids = set(sp.loc[sp.split == s, "image_id"])
        if not ids:
            errs.append(f"split {s} 이미지가 0장")
        elif ids - pool:
            errs.append(f"split {s} 중 LLVIP {name}에 없는 이미지 {len(ids - pool)}개")
    if errs:
        raise ValueError("split.csv 문제:\n  - " + "\n  - ".join(errs))
    return sp


def pick_val_segments(sp_a: pd.DataFrame, frac: float = 0.1, seed: int = 0) -> list:
    """A의 segment를 seed로 섞어, A 이미지의 frac 이상이 될 때까지 val로 뗀다 (학습용 segment는 최소 1개 남김)."""
    seg = sp_a.groupby("segment_id").size()
    if len(seg) < 2:
        raise ValueError("A의 segment가 2개 미만이라 segment 단위 val을 뗄 수 없음")
    order = seg.index.to_numpy()[np.random.default_rng(seed).permutation(len(seg))]
    target, val, n = frac * seg.sum(), [], 0
    for s in order[:-1]:
        if n >= target:
            break
        val.append(s)
        n += seg[s]
    return sorted(val)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split-csv", default=os.path.join(ROOT, "data", "split.csv"))
    ap.add_argument("--name", default="A_avg", help="models/<name>/ 에 저장")
    ap.add_argument("--model-tag", default="avg", help="후보 파일 이름의 모델 부분 (candidates_<tag>_B.csv)")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--val-seed", type=int, default=0)
    ap.add_argument("--plan", action="store_true", help="검사와 계획만 출력, 아무것도 쓰지 않음")
    ap.add_argument("--skip-train", action="store_true", help="학습을 건너뛰고 --weights로 후보만 저장")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--overwrite", action="store_true", help="이미 있는 후보 파일을 덮어씀")
    a = ap.parse_args()

    try:
        sp = read_split(a.split_csv)
    except (ValueError, FileNotFoundError) as e:
        sys.exit(str(e))
    ids = {s: sorted(sp.loc[sp.split == s, "image_id"]) for s in SPLITS}
    sp_a = sp[sp.split == "A"]
    val_segs = pick_val_segments(sp_a, a.val_frac, a.val_seed)
    val_ids = sorted(sp_a.loc[sp_a.segment_id.isin(val_segs), "image_id"])
    tr_ids = sorted(set(ids["A"]) - set(val_ids))
    assert not (set(val_ids) | set(tr_ids)) & (set(ids["B"]) | set(ids["test"]))

    cand_dir = os.path.join(ROOT, "candidates")
    out_b = os.path.join(cand_dir, f"candidates_{a.model_tag}_B.csv")
    out_t = os.path.join(cand_dir, f"DO_NOT_OPEN_candidates_{a.model_tag}_test.csv")
    run_dir = os.path.join(ROOT, "models", a.name)
    print(f"split.csv OK: A {len(ids['A'])} / B {len(ids['B'])} / test {len(ids['test'])} 장")
    print(f"A segment {sp_a.segment_id.nunique()}개 중 val {len(val_segs)}개 "
          f"(이미지 {len(val_ids)}장, {len(val_ids) / len(ids['A']):.1%}) -> 학습 {len(tr_ids)}장")
    print(f"학습 설정: yolo11n.pt {TRAIN_CFG} -> {run_dir}")
    print(f"후보: {out_b}\n      {out_t}")
    if a.plan:
        return

    for p in (out_b, out_t):
        if os.path.exists(p) and not a.overwrite:
            sys.exit(f"이미 있음: {p} (덮어쓰려면 --overwrite)")

    from fuse import fuse_avg
    from predict import predict_to_csv
    from prepare_proxy_data import write_split
    from ultralytics import YOLO

    fused = {s: os.path.join(ROOT, "data", "fused_avg", s) for s in SPLITS}
    fuse_avg("train", fused["A"], ids["A"])
    fuse_avg("train", fused["B"], ids["B"])
    fuse_avg("test", fused["test"], ids["test"])
    print("융합 완료:", {s: len(ids[s]) for s in SPLITS})

    if a.skip_train:
        if not a.weights:
            sys.exit("--skip-train 에는 --weights 가 필요")
        weights = a.weights
    else:
        yolo_dir = os.path.join(ROOT, "data", "yolo_A")
        for sub in ("images", "labels"):                 # 이전 실행의 val 구성이 남지 않게 새로 만듦
            shutil.rmtree(os.path.join(yolo_dir, sub), ignore_errors=True)
        print("train", write_split("train", tr_ids, fused["A"], yolo_dir))
        print("val", write_split("val", val_ids, fused["A"], yolo_dir))
        yaml = os.path.join(yolo_dir, "data.yaml")
        with open(yaml, "w") as f:
            f.write(f"path: {yolo_dir}\ntrain: images/train\nval: images/val\nnames:\n  0: person\n")
        os.makedirs(run_dir, exist_ok=True)
        pd.DataFrame({"segment_id": val_segs}).to_csv(os.path.join(run_dir, "A_val_segments.csv"), index=False)

        base = os.path.join(ROOT, "models", "yolo11n.pt")
        model = YOLO(base if os.path.exists(base) else "yolo11n.pt")
        model.train(data=yaml, project=os.path.join(ROOT, "models"), name=a.name, exist_ok=True, **TRAIN_CFG)
        weights = os.path.join(run_dir, "weights", "best.pt")

    model = YOLO(weights)
    for src, out in ((fused["B"], out_b), (fused["test"], out_t)):
        st = predict_to_csv(weights, src, out, model=model)
        warn = "  ⚠ max_det 1000에 닿은 이미지 있음" if st["n_at_max_det"] else ""
        print(f"후보 저장: {os.path.basename(out)} images={st['images']} cands={st['cands']} "
              f"per_image max={st['max']} n_at_1000={st['n_at_max_det']}{warn}")


if __name__ == "__main__":
    main()
