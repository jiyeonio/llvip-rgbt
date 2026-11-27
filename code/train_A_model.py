"""A로 탐지기 학습 (모델 이름을 인자로 받음). Test 이미지는 다루지 않는다.

학습 데이터·val 분리·학습 설정은 train_A.py와 같다 (read_split, pick_val_segments, TRAIN_CFG 재사용):
  학습 = split.csv A 중 val segment 제외, val = A 안에서 segment 단위(seed 0, 10% 이상),
  입력 = 평균 융합 이미지 data/fused_avg/A, imgsz 640, epochs 28, batch 16, seed 0, deterministic, workers 8.
다른 점: 베이스 가중치를 --base로 받는다. 융합은 A만 하고 Test는 건드리지 않는다.

python code/train_A_model.py --base yolo11s.pt --name A_avg_s --weights-out weights/A_avg_s_best.pt
"""
import argparse
import json
import os
import shutil
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval import ROOT  # noqa: E402
from train_A import TRAIN_CFG, pick_val_segments, read_split  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="COCO 사전학습 가중치 (yolo11n.pt / yolo11s.pt)")
    ap.add_argument("--name", required=True, help="models/<name>/ 에 저장")
    ap.add_argument("--weights-out", default=None, help="best.pt를 복사할 경로 (이미 있으면 멈춤)")
    ap.add_argument("--split-csv", default=os.path.join(ROOT, "data", "split.csv"))
    a = ap.parse_args()

    if a.weights_out and os.path.exists(a.weights_out):
        sys.exit(f"이미 있음: {a.weights_out}")
    sp = read_split(a.split_csv)
    sp_a = sp[sp.split == "A"]
    val_segs = pick_val_segments(sp_a, 0.1, 0)
    val_ids = sorted(sp_a.loc[sp_a.segment_id.isin(val_segs), "image_id"])
    tr_ids = sorted(set(sp_a.image_id) - set(val_ids))

    from fuse import fuse_avg
    from prepare_proxy_data import write_split
    from ultralytics import YOLO

    fused_a = os.path.join(ROOT, "data", "fused_avg", "A")
    fuse_avg("train", fused_a, sorted(sp_a.image_id))          # 이미 있으면 건너뜀
    yolo_dir = os.path.join(ROOT, "data", "yolo_A")
    for sub in ("images", "labels"):
        shutil.rmtree(os.path.join(yolo_dir, sub), ignore_errors=True)
    print("train", write_split("train", tr_ids, fused_a, yolo_dir))
    print("val", write_split("val", val_ids, fused_a, yolo_dir))
    yaml = os.path.join(yolo_dir, "data.yaml")
    with open(yaml, "w") as f:
        f.write(f"path: {yolo_dir}\ntrain: images/train\nval: images/val\nnames:\n  0: person\n")
    run_dir = os.path.join(ROOT, "models", a.name)
    os.makedirs(run_dir, exist_ok=True)
    pd.DataFrame({"segment_id": val_segs}).to_csv(os.path.join(run_dir, "A_val_segments.csv"), index=False)

    base = os.path.join(ROOT, "models", a.base)
    model = YOLO(base if os.path.exists(base) else a.base)
    print(f"학습: {a.base} {TRAIN_CFG} train {len(tr_ids)} / val {len(val_ids)} (segment {val_segs}) -> {run_dir}")
    t0 = time.time()
    model.train(data=yaml, project=os.path.join(ROOT, "models"), name=a.name, exist_ok=True, **TRAIN_CFG)
    wall = time.time() - t0

    res = pd.read_csv(os.path.join(run_dir, "results.csv"))
    best = os.path.join(run_dir, "weights", "best.pt")
    info = {"base": a.base, "name": a.name, "train_cfg": TRAIN_CFG, "n_train": len(tr_ids), "n_val": len(val_ids),
            "val_segments": [int(s) for s in val_segs], "wall_seconds": round(wall, 1),
            "results_csv_time_seconds": float(res["time"].iloc[-1]) if "time" in res else None,
            "best_mAP50_95": float(res["metrics/mAP50-95(B)"].max()), "best_pt": best}
    if a.weights_out:
        os.makedirs(os.path.dirname(os.path.abspath(a.weights_out)), exist_ok=True)
        shutil.copy2(best, a.weights_out)
        info["weights_out"] = a.weights_out
    with open(os.path.join(run_dir, "train_time.json"), "w") as f:
        json.dump(info, f, ensure_ascii=False, indent=1)
    print(json.dumps(info, ensure_ascii=False))


if __name__ == "__main__":
    main()
