"""E4 특징 함수 가짜 데이터 테스트."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval as E

box = np.array([[0, 0, 100, 200], [50, 0, 150, 200], [500, 500, 520, 540]], float)
conf = np.array([0.9, 0.5, 0.05])
g = E.geom_features(box, conf, np.arange(3))
assert np.allclose(g["max_ioa_ref"], [0.5, 0.5, 0]) and list(g["n_neigh"]) == [1, 1, 0]
assert list(g["rank_img"]) == [1, 2, 3]
assert np.allclose(g["dup_max_iou_higher"], [0, 1 / 3, 0]) and np.allclose(g["dup_conf_ratio"], [0, 0.5 / 0.9, 0])
assert list(g["dup_bundle_rank"]) == [1, 2, 1]
assert np.allclose(g["aspect"], [0.5, 0.5, 0.5]) and np.allclose(g["yc"], [100, 100, 520])
# 낮은 conf 후보만 특징 대상 + conf<0.1 후보는 ① 기준에서 제외, D 기준에는 포함
g2 = E.geom_features(box, np.array([0.05, 0.5, 0.9]), np.array([0]))
assert g2["max_ioa_ref"][0] == 0.5 and g2["dup_max_iou_higher"][0] > 0

img = np.full((1024, 1280), 100, np.uint8)
img[300:400, 200:250] = 200
img[300:333, 200:250] = 150                      # 위 1/3 정도 다른 밝기
b = np.array([[200, 300, 250, 400], [0.4, 0.6, 1279.5, 1023.2]])
f = E.ir_features(img, b)
assert f["ir_max"][0] == 200 and abs(f["ir_mean"][0] - img[300:400, 200:250].mean()) < 1e-9
r = 10                                            # 폭 50의 20%
ring = img[290:410, 190:260].astype(float).sum() - img[300:400, 200:250].astype(float).sum()
ring_mean = ring / (120 * 70 - 100 * 50)
assert abs(f["ir_ring_diff"][0] - (img[300:400, 200:250].mean() - ring_mean)) < 1e-9
assert abs(f["ir_top"][0] - img[300:333, 200:250].mean()) < 1e-9 and f["ir_bot"][0] == 200
assert f["ir_bright_frac"][0] == 1.0             # 80분위 = 100, 박스 안은 모두 100 초과
assert np.isnan(f["ir_ring_diff"][1])            # 이미지 전체 박스 -> 테두리 없음
assert abs(f["ir_mean"][1] - img.mean()) < 1e-9

rng = np.random.default_rng(0)
bgr = rng.integers(0, 255, (1024, 1280, 3), dtype=np.uint8)
import cv2
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(float)
fr = E.rgb_features(bgr, np.array([[10.2, 20.7, 60.1, 90.0]]))
patch = gray[20:90, 10:61]                       # 내림/올림
assert abs(fr["rgb_mean"][0] - patch.mean()) < 1e-6 and abs(fr["rgb_std"][0] - patch.std()) < 1e-4
print("all E4 feature tests ok")
