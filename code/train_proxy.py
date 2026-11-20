"""proxy 재학습: yolo11n.pt(COCO), imgsz 640, epochs 28, seed 0, 평균 융합 입력."""
import os
import sys

from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
batch = int(sys.argv[1]) if len(sys.argv) > 1 else 16

model = YOLO(os.path.join(ROOT, "models", "yolo11n.pt"))
model.train(data=os.path.join(ROOT, "data", "yolo_proxy", "data.yaml"),
            imgsz=640, epochs=28, seed=0, deterministic=True, batch=batch, workers=8,
            device=0, project=os.path.join(ROOT, "models"), name="proxy_retrain", exist_ok=True)
