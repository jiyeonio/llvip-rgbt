"""모델로 후보 생성 -> candidates/candidates_{model}_{split}.csv

python code/predict.py <weights.pt> <image_dir> <out_csv>
설정: conf=0.001, iou=0.7, imgsz=640, max_det=1000, stream=True
"""
import os
import sys

import pandas as pd

MAX_DET = 1000


def predict_to_csv(weights: str, src: str, out: str, model=None) -> dict:
    """src 폴더 이미지에 예측해 후보 CSV 저장. 이미지당 후보 수 통계를 반환 (내용은 출력하지 않음)."""
    from ultralytics import YOLO
    model = model or YOLO(weights)
    rows, per_img = [], {}
    for r in model.predict(source=src, conf=0.001, iou=0.7, imgsz=640, max_det=MAX_DET,
                           stream=True, verbose=False, device=0):
        iid = os.path.splitext(os.path.basename(r.path))[0]
        xyxy, conf = r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()
        per_img[iid] = len(conf)
        for (x1, y1, x2, y2), c in zip(xyxy, conf):
            rows.append((iid, float(x1), float(y1), float(x2), float(y2), float(c)))
    df = pd.DataFrame(rows, columns=["image_id", "x1", "y1", "x2", "y2", "conf"])
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    df.to_csv(out, index=False, float_format="%.6f")
    s = pd.Series(per_img, dtype=int)
    return dict(images=len(s), cands=len(df), mean=float(s.mean()) if len(s) else 0.0,
                max=int(s.max()) if len(s) else 0, n_at_max_det=int((s >= MAX_DET).sum()), out=out)


if __name__ == "__main__":
    weights, src, out = sys.argv[1:4]
    st = predict_to_csv(weights, src, out)
    print(f"images={st['images']} cands={st['cands']} per_image mean={st['mean']:.1f} max={st['max']} "
          f"n_at_1000={st['n_at_max_det']} -> {out}")
