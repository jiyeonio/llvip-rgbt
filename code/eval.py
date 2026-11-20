"""LLVIP 공통 평가 함수.

정의:
- 매칭: conf 높은 순으로 1:1, IoU >= 0.5 (각 후보는 아직 짝 없는 정답 중 IoU 최대인 것과 짝)
- TP / FP: 짝이 생긴 후보 / 짝이 없는 후보
- FPPI: FP 수 / 이미지 수 (정답이 없는 이미지도 분모에 포함)
- Q1(정답): 원본 높이 184px 미만
- 겹침(정답): 다른 정답과 IoA > 0.3 (IoA = 내 박스 넓이 중 겹친 비율)

경로는 아래 상수 또는 환경변수로 지정한다.
"""
from __future__ import annotations

import glob
import os
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- 경로 설정 ---
# LLVIP 배포본은 Annotations/ 한 폴더에 train+test XML이 섞여 있다.
# split 소속은 visible/{train,test}/ 의 이미지 파일 이름으로 정한다.
LLVIP_DIR = os.environ.get("LLVIP_DIR", os.path.join(ROOT, "data", "LLVIP"))
XML_DIR = os.path.join(LLVIP_DIR, "Annotations")
SPLIT_CSV = os.environ.get("LLVIP_SPLIT_CSV", os.path.join(ROOT, "data", "split.csv"))
RESULTS_DIR = os.environ.get("LLVIP_RESULTS", os.path.join(ROOT, "results"))  # 테스트 시 다른 폴더로


def llvip_ids(subset: str) -> list[str]:
    """subset: "train" / "test" -> visible/{subset}/ 의 image_id 정렬 목록."""
    d = os.path.join(LLVIP_DIR, "visible", subset)
    return sorted(os.path.splitext(f)[0] for f in os.listdir(d) if f.endswith(".jpg"))

IMG_W, IMG_H = 1280, 1024
IOU_THR = 0.5
Q1_HEIGHT = 184          # 원본 기준 (640 기준 92px)
OVERLAP_IOA = 0.3
CAND_COLS = ["image_id", "x1", "y1", "x2", "y2", "conf"]


# ---------------------------------------------------------------- 기하
def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (N,4), b: (M,4) xyxy -> (N,M) IoU."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def ioa_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N,M): a[i] 넓이 중 b[j]와 겹친 비율."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    return inter / np.maximum(area_a[:, None], 1e-12)


def overlap_flags(boxes: np.ndarray, others: np.ndarray | None = None,
                  same_set: bool = True, thr: float = OVERLAP_IOA) -> np.ndarray:
    """각 박스가 다른 박스(others)와 IoA > thr 인지. same_set이면 자기 자신 제외."""
    others = boxes if others is None else others
    m = ioa_matrix(boxes, others)
    if same_set and len(boxes):
        np.fill_diagonal(m, 0.0)
    return (m > thr).any(axis=1) if m.size else np.zeros(len(boxes), bool)


# ---------------------------------------------------------------- 로딩
def _parse_xml(path: str) -> list[tuple[float, float, float, float]]:
    root = ET.parse(path).getroot()
    out = []
    for obj in root.iter("object"):
        bb = obj.find("bndbox")
        out.append(tuple(float(bb.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax")))
    return out


def load_gt(split: str) -> pd.DataFrame:
    """정답 박스 로드.

    split: "test" -> LLVIP test 전체, "train" -> LLVIP train 전체,
           그 외("B" 등) -> LLVIP train 중 split.csv에서 해당 split 인 이미지.
    반환: DataFrame[image_id, x1, y1, x2, y2, height, q1, overlap]
          df.attrs["image_ids"] = 평가 대상 이미지 전체 목록 (정답 0개 이미지 포함, FPPI 분모)
    """
    if split.lower() in ("test", "train"):
        image_ids = llvip_ids(split.lower())
    else:
        sp = pd.read_csv(SPLIT_CSV, dtype={"image_id": str})
        image_ids = sorted(sp.loc[sp["split"] == split, "image_id"].str.zfill(6))
        if not image_ids:
            raise ValueError(f"split.csv에 split={split!r} 이미지가 없음")

    rows = []
    for iid in image_ids:
        p = os.path.join(XML_DIR, f"{iid}.xml")
        if not os.path.exists(p):
            raise FileNotFoundError(f"XML 없음: {p}")
        for b in _parse_xml(p):
            rows.append((iid, *b))

    gt = pd.DataFrame(rows, columns=["image_id", "x1", "y1", "x2", "y2"])
    return add_gt_flags(gt, image_ids)


def add_gt_flags(gt: pd.DataFrame, image_ids) -> pd.DataFrame:
    """정답 DataFrame에 height / q1 / overlap 열을 붙인다."""
    gt = gt.reset_index(drop=True).copy()
    gt["image_id"] = gt["image_id"].astype(str)
    gt["height"] = gt["y2"] - gt["y1"]
    gt["q1"] = gt["height"] < Q1_HEIGHT
    ov = np.zeros(len(gt), bool)
    for _, idx in gt.groupby("image_id").indices.items():
        ov[idx] = overlap_flags(gt.loc[idx, ["x1", "y1", "x2", "y2"]].to_numpy(float))
    gt["overlap"] = ov
    gt.attrs["image_ids"] = sorted(set(map(str, image_ids)))
    return gt


def load_candidates(path: str) -> pd.DataFrame:
    """후보 csv 로드 (image_id는 6자리 문자열 유지)."""
    df = pd.read_csv(path, dtype={"image_id": str})
    missing = [c for c in CAND_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: 열 누락 {missing}")
    df = df[CAND_COLS].copy()
    df["image_id"] = df["image_id"].str.zfill(6)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------- 매칭
def _match_image(cb: np.ndarray, gb: np.ndarray, iou_thr: float):
    """cb: conf 내림차순 정렬된 후보 박스. 반환 (cand_gt_idx, gt_matched)."""
    cand_gt = np.full(len(cb), -1, int)
    gt_used = np.zeros(len(gb), bool)
    if len(cb) == 0 or len(gb) == 0:
        return cand_gt, gt_used
    ious = iou_matrix(cb, gb)
    for i in range(len(cb)):
        row = np.where(gt_used, -1.0, ious[i])
        j = int(row.argmax())
        if row[j] >= iou_thr:
            cand_gt[i] = j
            gt_used[j] = True
            if gt_used.all():
                break
    return cand_gt, gt_used


def match(cands: pd.DataFrame, gts: pd.DataFrame, thr: float, iou_thr: float = IOU_THR) -> dict:
    """conf >= thr 후보를 conf 높은 순으로 정답과 1:1 매칭.

    반환 dict:
      tp, fp, fn          : 개수
      cand_tp             : 후보별 TP 여부 (cands 인덱스 기준 Series, conf < thr 후보는 False)
      cand_gt             : 후보별 짝 정답의 gts 인덱스 (없으면 -1)
      gt_matched          : 정답별 매칭 여부 (gts 인덱스 기준 Series)

    참고: 탐욕 매칭은 conf 순서로 진행하므로, thr 이상 후보만의 매칭 결과는
    전체 후보 매칭에서 conf >= thr 인 부분과 같다 (동점 conf 제외). E1/E3 스윕에서 이용.
    """
    sel = cands[cands["conf"] >= thr]
    cand_tp = pd.Series(False, index=cands.index)
    cand_gt = pd.Series(-1, index=cands.index)
    gt_matched = pd.Series(False, index=gts.index)

    g_groups = gts.groupby("image_id").indices
    gbox_all = gts[["x1", "y1", "x2", "y2"]].to_numpy(float)
    gidx_all = gts.index.to_numpy()
    for iid, cidx in sel.groupby("image_id").indices.items():
        sub = sel.iloc[cidx]
        order = np.argsort(-sub["conf"].to_numpy(), kind="stable")
        sub = sub.iloc[order]
        gpos = g_groups.get(iid, np.array([], int))
        cg, gu = _match_image(sub[["x1", "y1", "x2", "y2"]].to_numpy(float), gbox_all[gpos], iou_thr)
        hit = cg >= 0
        cand_tp.loc[sub.index[hit]] = True
        cand_gt.loc[sub.index[hit]] = gidx_all[gpos[cg[hit]]]
        gt_matched.loc[gidx_all[gpos[gu]]] = True

    tp = int(cand_tp.sum())
    return dict(tp=tp, fp=int(len(sel) - tp), fn=int((~gt_matched).sum()),
                cand_tp=cand_tp, cand_gt=cand_gt, gt_matched=gt_matched)


def _n_images(gts: pd.DataFrame, cands: pd.DataFrame | None = None) -> int:
    ids = gts.attrs.get("image_ids")
    if ids is None:
        ids = set(gts["image_id"]) | (set(cands["image_id"]) if cands is not None else set())
    return len(ids)


def fppi_recall(cands: pd.DataFrame, gts: pd.DataFrame, thr: float, m: dict | None = None) -> dict:
    """conf >= thr 에서 FPPI와 Recall (전체 / Q1 / 겹침)."""
    m = m or match(cands, gts, thr)
    gm = m["gt_matched"]
    rec = lambda mask: float(gm[mask].mean()) if mask.any() else float("nan")
    return dict(thr=thr, n_images=_n_images(gts, cands), tp=m["tp"], fp=m["fp"], fn=m["fn"],
                fppi=m["fp"] / _n_images(gts, cands),
                recall=rec(np.ones(len(gts), bool)),
                recall_q1=rec(gts["q1"].to_numpy()),
                recall_overlap=rec(gts["overlap"].to_numpy()),
                n_gt=len(gts), n_q1=int(gts["q1"].sum()), n_overlap=int(gts["overlap"].sum()))


def miss_causes(cands: pd.DataFrame, gts: pd.DataFrame, thr: float = 0.25) -> pd.DataFrame:
    """정답별 원인: hit / low_conf / no_candidate.

    hit         : conf >= thr 후보와 1:1 매칭됨
    low_conf    : hit 아님, 저장된 후보(모든 conf) 중 IoU >= 0.5 인 후보가 있음
    no_candidate: IoU >= 0.5 후보가 하나도 없음
    반환: gts 에 'cause' 열을 붙인 DataFrame
    """
    hit = match(cands, gts, thr)["gt_matched"].to_numpy()
    any_cand = np.zeros(len(gts), bool)
    c_groups = cands.groupby("image_id").indices
    cbox_all = cands[["x1", "y1", "x2", "y2"]].to_numpy(float)
    for iid, gpos in gts.groupby("image_id").indices.items():
        cpos = c_groups.get(iid)
        if cpos is None:
            continue
        ious = iou_matrix(gts.iloc[gpos][["x1", "y1", "x2", "y2"]].to_numpy(float), cbox_all[cpos])
        any_cand[gpos] = (ious >= IOU_THR).any(axis=1)
    out = gts.copy()
    out["cause"] = np.where(hit, "hit", np.where(any_cand, "low_conf", "no_candidate"))
    return out


def miss_cause_table(causes: pd.DataFrame) -> pd.DataFrame:
    """전체 / Q1 / 겹침 그룹별 원인 비율."""
    rows = []
    for name, mask in [("all", np.ones(len(causes), bool)),
                       ("q1", causes["q1"].to_numpy()), ("overlap", causes["overlap"].to_numpy())]:
        c = causes.loc[mask, "cause"]
        rows.append({"group": name, "n_gt": len(c),
                     **{k: float((c == k).mean()) if len(c) else float("nan")
                        for k in ("hit", "low_conf", "no_candidate")}})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # 1번 검증: proxy test 후보, conf 0.25 -> 전체 0.890 / Q1 0.733 / 겹침 0.757 (±0.002)
    path = os.path.join(ROOT, "candidates", "candidates_proxy_test.csv")
    gts = load_gt("test")
    cands = load_candidates(path)
    r = fppi_recall(cands, gts, 0.25)
    print(r)
    target = dict(recall=0.890, recall_q1=0.733, recall_overlap=0.757)
    ok = all(abs(r[k] - v) <= 0.002 for k, v in target.items())
    print("검증", "통과" if ok else "실패", {k: round(r[k] - v, 4) for k, v in target.items()})


# ---------------------------------------------------------------- 스윕 / 결과 표시 도우미
def sweep(cands: pd.DataFrame, gts: pd.DataFrame, thrs) -> pd.DataFrame:
    """여러 임계값의 FPPI/Recall을 한 번의 매칭으로 계산.

    conf 순서 탐욕 매칭은 prefix 성질이 있어(match 참고), 최저 임계값에서 한 번 매칭한 뒤
    conf >= thr 부분만 세면 각 임계값에서 직접 매칭한 결과와 같다.
    """
    thrs = np.sort(np.asarray(thrs, float))
    m = match(cands, gts, float(thrs[0]))
    conf = cands["conf"].to_numpy()
    sel = conf >= thrs[0]
    tp_conf = np.sort(conf[m["cand_tp"].to_numpy()])
    all_conf = np.sort(conf[sel])
    # 정답별: 짝 후보의 conf (없으면 -inf)
    gt_conf = np.full(len(gts), -np.inf)
    ct = m["cand_gt"]
    hit = ct >= 0
    gt_conf[gts.index.get_indexer(ct[hit].to_numpy())] = conf[hit.to_numpy()]
    n_img = _n_images(gts, cands)
    q1, ov = gts["q1"].to_numpy(), gts["overlap"].to_numpy()
    rows = []
    for t in thrs:
        tp = len(tp_conf) - np.searchsorted(tp_conf, t, "left")
        n = len(all_conf) - np.searchsorted(all_conf, t, "left")
        g = gt_conf >= t
        rows.append(dict(thr=t, tp=tp, fp=n - tp, fppi=(n - tp) / n_img, recall=g.mean(),
                         recall_q1=g[q1].mean(), recall_overlap=g[ov].mean()))
    return pd.DataFrame(rows)


def recall_at_fppi(curve: pd.DataFrame, target: float, col: str = "recall") -> float:
    """FPPI target에서 Recall: 가장 가까운 두 점(target 이하 최대 FPPI, 이상 최소 FPPI) 사이 선형 보간."""
    c = curve.sort_values(["fppi", col])
    lo, hi = c[c.fppi <= target], c[c.fppi >= target]
    if lo.empty or hi.empty:
        return float("nan")
    a, b = lo.iloc[-1], hi.iloc[0]
    if b.fppi == a.fppi:
        return float(max(a[col], b[col]))
    return float(a[col] + (b[col] - a[col]) * (target - a.fppi) / (b.fppi - a.fppi))


def parse_cand_name(path: str) -> tuple[str, str]:
    """candidates_{model}_{split}.csv -> (model, split)."""
    stem = os.path.splitext(os.path.basename(path))[0]
    _, model, split = stem.split("_", 2)
    return model, split


PROXY_NOTE_CSV = "코드 검증용(proxy 재학습, Test)"
PROXY_NOTE_PNG = "proxy 재학습 · Test · 코드 검증용"


def use_korean_font():
    """matplotlib 한글 폰트(나눔고딕, koreanize-matplotlib 패키지 동봉) 설정."""
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import koreanize_matplotlib  # noqa: F401  (폰트 등록)
    fdir = os.path.join(os.path.dirname(koreanize_matplotlib.__file__), "fonts")
    for f in glob.glob(os.path.join(fdir, "*.ttf")):
        font_manager.fontManager.addfont(f)
    plt.rcParams["font.family"] = "NanumGothic"
    plt.rcParams["axes.unicode_minus"] = False


def save_csv(df: pd.DataFrame, path: str, model: str, note: str | None = None):
    """결과 csv 저장. proxy면 첫 줄에 검증용 표시, note가 있으면 그다음 줄(또는 첫 줄)에 표시."""
    with open(path, "w", encoding="utf-8") as f:
        if model == "proxy":
            f.write(PROXY_NOTE_CSV + "\n")
        if note:
            f.write(note + "\n")
        df.to_csv(f, index=False, float_format="%.4f")
