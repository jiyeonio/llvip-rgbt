"""n / s 추론 속도 비교 (B 이미지만). results/yolo_s/speed.csv

같은 GPU, imgsz 640, batch 1, 후보 생성과 같은 설정(conf 0.001, iou 0.7, max_det 1000).
B 평균 융합 이미지를 파일 이름 순으로 정렬해 고르게 500장 고르고(np.linspace), 다른 30장으로 warmup.
측정: ultralytics가 재는 이미지당 전처리·추론·후처리 시간(ms)의 평균·중앙값, 이미지 읽기 포함 wall time.
n → s → n → s 두 번 돌려 두 번째 값을 쓴다 (순서 효과 확인용으로 첫 번째도 저장).

python code/speed_yolo.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval import RESULTS_DIR, ROOT  # noqa: E402

MODELS = {"YOLO11n": os.path.join(ROOT, "weights", "A_avg_best.pt"),
          "YOLO11s": os.path.join(ROOT, "weights", "A_avg_s_best.pt")}
SRC = os.path.join(ROOT, "data", "fused_avg", "B")
N_TIMED, N_WARM = 500, 30


def main():
    import torch
    from ultralytics import YOLO
    files = sorted(os.path.join(SRC, f) for f in os.listdir(SRC) if f.endswith(".jpg"))
    idx = np.unique(np.linspace(0, len(files) - 1, N_TIMED).round().astype(int))
    timed = [files[i] for i in idx]
    rest = [f for i, f in enumerate(files) if i not in set(idx)]
    warm = rest[:: max(1, len(rest) // N_WARM)][:N_WARM]
    assert len(timed) == N_TIMED and not set(timed) & set(warm)
    gpu = torch.cuda.get_device_name(0)

    rows = []
    for rep in (1, 2):
        for name, w in MODELS.items():
            model = YOLO(w)
            from ultralytics.utils.torch_utils import get_flops
            params = sum(x.numel() for x in model.model.parameters())
            gflops = get_flops(model.model, imgsz=640)
            kw = dict(conf=0.001, iou=0.7, imgsz=640, max_det=1000, batch=1, device=0, verbose=False)
            for f in warm:
                model.predict(f, **kw)
            pre, inf, post, wall = [], [], [], []
            for f in timed:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                r = model.predict(f, **kw)[0]
                torch.cuda.synchronize()
                wall.append((time.perf_counter() - t0) * 1e3)
                pre.append(r.speed["preprocess"]); inf.append(r.speed["inference"]); post.append(r.speed["postprocess"])
            tot = np.array(pre) + np.array(inf) + np.array(post)
            rows.append({"repeat": rep, "model": name, "weights": os.path.relpath(w, ROOT), "gpu": gpu,
                         "imgsz": 640, "batch": 1, "n_images": len(timed), "params_M": params / 1e6, "GFLOPs": gflops,
                         "inference_ms_mean": np.mean(inf), "inference_ms_median": np.median(inf),
                         "pre+inf+post_ms_mean": tot.mean(), "pre+inf+post_ms_median": np.median(tot),
                         "wall_ms_mean(읽기 포함)": np.mean(wall), "wall_ms_median(읽기 포함)": np.median(wall),
                         "preprocess_ms_mean": np.mean(pre), "postprocess_ms_mean": np.mean(post)})
            print(rows[-1])
            del model
            torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    os.makedirs(os.path.join(RESULTS_DIR, "yolo_s"), exist_ok=True)
    df.to_csv(os.path.join(RESULTS_DIR, "yolo_s", "speed.csv"), index=False, float_format="%.3f")


if __name__ == "__main__":
    main()
