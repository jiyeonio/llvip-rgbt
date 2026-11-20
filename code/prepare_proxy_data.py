"""proxy 재학습용 YOLO 데이터셋 준비.

train: LLVIP Train 파일 이름 정렬 후 [::4] (3,007쌍), 평균 융합 이미지
val  : 학습에 안 쓴 Train 프레임 [2::4][::8] (best.pt 선택용, Test는 쓰지 않음)
라벨 : VOC XML -> YOLO (person 1클래스, 0~1 정규화, 이미지 밖은 잘라냄)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval import ROOT, XML_DIR, IMG_W, IMG_H, _parse_xml, llvip_ids  # noqa: E402
from fuse import fuse_avg  # noqa: E402

OUT = os.path.join(ROOT, "data", "yolo_proxy")


def write_split(name, ids, fused_dir, out=OUT):
    """YOLO 데이터셋 한 split 작성: out/images/{name}/ (융합 이미지 symlink), out/labels/{name}/ (라벨)."""
    img_d, lab_d = os.path.join(out, "images", name), os.path.join(out, "labels", name)
    os.makedirs(img_d, exist_ok=True)
    os.makedirs(lab_d, exist_ok=True)
    nbox = 0
    for iid in ids:
        link = os.path.join(img_d, f"{iid}.jpg")
        if not os.path.lexists(link):
            os.symlink(os.path.join(fused_dir, f"{iid}.jpg"), link)
        lines = []
        for x1, y1, x2, y2 in _parse_xml(os.path.join(XML_DIR, f"{iid}.xml")):
            x1, x2 = max(0, x1), min(IMG_W, x2)
            y1, y2 = max(0, y1), min(IMG_H, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            lines.append(f"0 {(x1+x2)/2/IMG_W:.6f} {(y1+y2)/2/IMG_H:.6f} {(x2-x1)/IMG_W:.6f} {(y2-y1)/IMG_H:.6f}")
        nbox += len(lines)
        with open(os.path.join(lab_d, f"{iid}.txt"), "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
    return len(ids), nbox


if __name__ == "__main__":
    ids = llvip_ids("train")
    tr, va = ids[::4], ids[2::4][::8]
    assert not set(tr) & set(va)
    fused_tr = os.path.join(ROOT, "data", "fused_avg", "train")
    fused_va = os.path.join(ROOT, "data", "fused_avg", "train_val")
    fuse_avg("train", fused_tr, tr)
    fuse_avg("train", fused_va, va)
    print("train", write_split("train", tr, fused_tr))
    print("val", write_split("val", va, fused_va))
    with open(os.path.join(OUT, "data.yaml"), "w") as f:
        f.write(f"path: {OUT}\ntrain: images/train\nval: images/val\nnames:\n  0: person\n")
    print("->", os.path.join(OUT, "data.yaml"))
