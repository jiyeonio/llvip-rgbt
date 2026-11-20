"""RGB-열화상 평균 융합 (0.5:0.5 픽셀 평균).

사용:
  python code/fuse.py test                      -> data/fused_avg/test/
  python code/fuse.py train --every 4           -> data/fused_avg/train/ (정렬 후 4장마다 1장)
  (B는 split.csv가 생기면 fuse_avg("train", out_dir, ids=B_ids)로 호출)
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LLVIP_DIR = os.environ.get("LLVIP_DIR", os.path.join(ROOT, "data", "LLVIP"))
JPEG_QUALITY = 95


def fuse_pair(vis_path: str, ir_path: str) -> np.ndarray:
    v = cv2.imread(vis_path, cv2.IMREAD_COLOR)
    r = cv2.imread(ir_path, cv2.IMREAD_COLOR)
    if v is None or r is None:
        raise FileNotFoundError(f"이미지 읽기 실패: {vis_path} / {ir_path}")
    if v.shape != r.shape:
        raise ValueError(f"크기 불일치: {vis_path} {v.shape} vs {r.shape}")
    # (v + r) / 2, 반올림
    return ((v.astype(np.uint16) + r.astype(np.uint16) + 1) // 2).astype(np.uint8)


def _job(args):
    iid, subset, out_dir = args
    out = os.path.join(out_dir, f"{iid}.jpg")
    if os.path.exists(out):
        return
    img = fuse_pair(os.path.join(LLVIP_DIR, "visible", subset, f"{iid}.jpg"),
                    os.path.join(LLVIP_DIR, "infrared", subset, f"{iid}.jpg"))
    cv2.imwrite(out, img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])


def fuse_avg(subset: str, out_dir: str, ids: list[str] | None = None, workers: int = 8) -> int:
    """subset("train"/"test")의 이미지(ids 지정 시 그것만)를 평균 융합해 out_dir/{id}.jpg로 저장."""
    if ids is None:
        d = os.path.join(LLVIP_DIR, "visible", subset)
        ids = sorted(os.path.splitext(f)[0] for f in os.listdir(d) if f.endswith(".jpg"))
    os.makedirs(out_dir, exist_ok=True)
    with ProcessPoolExecutor(workers) as ex:
        list(ex.map(_job, [(i, subset, out_dir) for i in ids], chunksize=32))
    return len(ids)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("subset", choices=["train", "test"])
    ap.add_argument("--every", type=int, default=1, help="정렬 후 N장마다 1장")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d = os.path.join(LLVIP_DIR, "visible", a.subset)
    ids = sorted(os.path.splitext(f)[0] for f in os.listdir(d) if f.endswith(".jpg"))[::a.every]
    out = a.out or os.path.join(ROOT, "data", "fused_avg", a.subset)
    print(a.subset, fuse_avg(a.subset, out, ids), "->", out)
