"""r2_chaerin.py — 1번 묶음 (재학습 없음): 시드 앙상블 / conf×확률^α / soft-NMS 중복 억제.

r1_chaerin.py 로 만든 체크포인트(results/R0_1to1/seed{0,1,2}.pt)를 불러서 '점수 만드는 방식'만 바꾼다.

    python hackathon/members/r2_chaerin.py smoke    # 점검(기록 안 됨): α=1 이 R7-a 시드0 과 같은지, soft-NMS 동작
    python hackathon/members/r2_chaerin.py batch1   # 6개 실험을 순서대로 리더보드에 기록
    python hackathon/members/r2_chaerin.py X4       # ID 가 X4 로 시작하는 실험만 기록
"""
import os
import sys
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OWNER = "채린"

# ----------------------------------------------------------------------------- r1 지연 import
_R = None


def R():
    """r1_chaerin 을 처음 필요할 때 불러온다 (체크포인트 경로·Crops·build 를 그대로 재사용)."""
    global _R
    if _R is None:
        import r1_chaerin
        r1_chaerin.MODE = "r7a"          # ckpt_path 가 seed{n}.pt 를 가리키게 함
        _R = r1_chaerin
    return _R


# ----------------------------------------------------------------------------- 순수 함수 (모델 불필요)
def _iou_matrix(b):
    x1 = np.maximum(b[:, None, 0], b[None, :, 0])
    y1 = np.maximum(b[:, None, 1], b[None, :, 1])
    x2 = np.minimum(b[:, None, 2], b[None, :, 2])
    y2 = np.minimum(b[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area[:, None] + area[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def combine(conf, mask, p, alpha=1.0):
    """재점수 대상(mask)만 conf × p^alpha 로 바꾼다. 나머지는 그대로."""
    new = np.array(conf, dtype=float, copy=True)
    new[mask] = new[mask] * np.power(p, alpha)
    return new


def suppress(df, mask, kind, t):
    """중복 억제. 재점수 대상 박스마다, 같은 이미지에서 자기보다 conf 가 높은 박스와의 최대 IoU(mi)를 구해
    점수를 깎는다 (높은 점수 박스 자체는 건드리지 않음).
      lin   : mi >= t 이면 점수 × (1 - mi)          (선형 soft-NMS, 임계 t)
      gauss : 점수 × exp(-mi² / t)                  (가우시안 soft-NMS, σ=t)
      hard  : mi >= t 이면 점수 × 0.001             (사실상 제거)
    간단화: 깎인 점수를 다시 순서에 반영하지 않고 원래 conf 순서로 한 번만 계산한다."""
    conf = df["conf"].to_numpy(float)
    new = conf.copy()
    boxes = df[["x1", "y1", "x2", "y2"]].to_numpy(float)
    mask = np.asarray(mask, bool)
    for _, idx in df.groupby("image_id").indices.items():
        idx = np.asarray(idx)
        m = mask[idx]
        if not m.any():
            continue
        c = conf[idx]
        iou = _iou_matrix(boxes[idx])
        np.fill_diagonal(iou, 0.0)
        pos = np.empty(len(idx), int)
        pos[np.argsort(-c, kind="stable")] = np.arange(len(idx))
        higher = pos[None, :] < pos[:, None]          # higher[i, j] = j 가 i 보다 위
        mi = np.where(higher, iou, 0.0).max(axis=1)
        if kind == "lin":
            pen = np.where(mi >= t, 1.0 - mi, 1.0)
        elif kind == "gauss":
            pen = np.exp(-(mi ** 2) / t)
        elif kind == "hard":
            pen = np.where(mi >= t, 1e-3, 1.0)
        else:
            raise ValueError(f"모르는 억제 방식: {kind}")
        new[idx[m]] = c[m] * pen[m]
    return new


# ----------------------------------------------------------------------------- 모델 확률 (시드별 캐시)
_P = {}


def seed_probs(env, cands, seed):
    """(mask, p): B 후보 중 재점수 대상의 ResNet 확률. 같은 시드는 한 번만 계산해서 재사용."""
    key = (seed, len(cands))
    if key in _P:
        return _P[key]
    import torch
    from torch.utils.data import DataLoader
    r = R()
    if hasattr(env, "llvip_dir"):
        r.RC.ROOT = env.llvip_dir
    path = r.ckpt_path(seed)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} 없음. 먼저 r1_chaerin.py full 로 체크포인트를 만드세요")
    model = r.build(pretrain=False).to(env.device)
    model.load_state_dict(torch.load(path, map_location=env.device))
    model.eval()
    mask = np.asarray(env.rescore_mask(cands), bool)
    print(f"  [시드 {seed}] ResNet 확률 계산 · {int(mask.sum()):,}개", flush=True)
    dl = DataLoader(r.Crops(cands[mask].reset_index(drop=True)), batch_size=512, shuffle=False, num_workers=4)
    with torch.no_grad():
        p = np.concatenate([torch.sigmoid(model(x.to(env.device)).squeeze(1)).float().cpu().numpy() for x, _ in dl])
    _P[key] = (mask, p)
    return _P[key]


def train_model(env, seed):
    """재학습 없음: 체크포인트만 불러온다 (파라미터 수 기록용으로 모델을 state 에 넣어 둠)."""
    import torch
    r = R()
    model = r.build(pretrain=False).to(env.device)
    model.load_state_dict(torch.load(r.ckpt_path(seed), map_location=env.device))
    return {"seed": seed, "model": model}


def train_none(env, seed):
    return {"seed": seed}


def make_rescore(kind, **kw):
    def rescore(state, cands, env):
        conf = cands["conf"].to_numpy(float)
        if kind == "alpha":
            mask, p = seed_probs(env, cands, state["seed"])
            return combine(conf, mask, p, kw["alpha"])
        if kind == "ens":
            ps = [seed_probs(env, cands, s) for s in kw["seeds"]]
            return combine(conf, ps[0][0], np.mean([x[1] for x in ps], axis=0), kw.get("alpha", 1.0))
        if kind == "snms":
            return suppress(cands, env.rescore_mask(cands), kw["mode"], kw["t"])
        raise ValueError(kind)
    return rescore


# ----------------------------------------------------------------------------- 실험 목록
S3 = (0, 1, 2)
CACHE_NOTE = "추론 시간(ms)은 확률 캐시 때문에 실제보다 짧게 나옴"
DET_NOTE = "학습·무작위 없음(결정적)이라 시드 1회. σ 없음"
EXPERIMENTS = [
    dict(id="X1-ens3", change="점수 결합 (앙상블)", train=train_model, seeds=(0,), note=CACHE_NOTE + " · " + DET_NOTE,
         setting="R0-1to1 시드 3개 확률 평균 → YOLO conf × 평균 확률 (재학습 없음)", kind="ens", kw=dict(seeds=S3)),
    dict(id="X2-a0.5", change="점수 결합 (α)", train=train_model, seeds=S3, note=CACHE_NOTE,
         setting="YOLO conf × ResNet 확률^0.5 (R0-1to1 시드 3개 각각, 재학습 없음)", kind="alpha", kw=dict(alpha=0.5)),
    dict(id="X3-a2.0", change="점수 결합 (α)", train=train_model, seeds=S3, note=CACHE_NOTE,
         setting="YOLO conf × ResNet 확률^2 (R0-1to1 시드 3개 각각, 재학습 없음)", kind="alpha", kw=dict(alpha=2.0)),
    dict(id="X4-snms-lin0.5", change="중복 억제", train=train_none, seeds=(0,), note=DET_NOTE,
         setting="ResNet 없음. 재점수 대상 박스를 더 높은 conf 박스와의 IoU≥0.5 이면 ×(1-IoU) 로 깎음 (선형 soft-NMS)",
         kind="snms", kw=dict(mode="lin", t=0.5)),
    dict(id="X5-snms-lin0.3", change="중복 억제", train=train_none, seeds=(0,), note=DET_NOTE,
         setting="ResNet 없음. IoU≥0.3 이면 ×(1-IoU) (선형 soft-NMS, 더 공격적)", kind="snms", kw=dict(mode="lin", t=0.3)),
    dict(id="X6-snms-gauss0.5", change="중복 억제", train=train_none, seeds=(0,), note=DET_NOTE,
         setting="ResNet 없음. ×exp(-IoU²/0.5) (가우시안 soft-NMS)", kind="snms", kw=dict(mode="gauss", t=0.5)),
]


def run(exp, lb, dry_run=False):
    print(f"\n######## {exp['id']} ({exp['change']}) ########", flush=True)
    lb.submit(exp["id"], OWNER, exp["change"], exp["setting"], exp["train"], make_rescore(exp["kind"], **exp["kw"]),
              seeds=exp["seeds"], note=exp.get("note", ""), dry_run=dry_run)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode not in ("smoke", "batch1") and not any(e["id"].startswith(mode) for e in EXPERIMENTS if mode):
        sys.exit("사용: python r2_chaerin.py [smoke | batch1 | X1 | X2 | X3 | X4 | X5 | X6]")
    lb = R().lb
    if mode == "smoke":
        print("점검 1: α=1 은 R7-a-1to1 시드0 과 같아야 함 (Q1 0.700 / 겹침 0.779 / 전체 0.886, ΔQ1 +1.50%p)")
        lb.submit("X0-check", OWNER, "점검", "α=1 재현 점검", train_model, make_rescore("alpha", alpha=1.0),
                  seeds=(0,), dry_run=True)
        print("\n점검 2: soft-NMS 가 돌아가고 숫자가 나오는지")
        lb.submit("X0-check2", OWNER, "점검", "soft-NMS 점검", train_none,
                  make_rescore("snms", mode="lin", t=0.5), seeds=(0,), dry_run=True)
        return
    todo = EXPERIMENTS if mode == "batch1" else [e for e in EXPERIMENTS if e["id"].startswith(mode)]
    failed = []
    for e in todo:
        try:
            run(e, lb)
        except Exception:
            traceback.print_exc()
            failed.append(e["id"])
    print("\n완료." + (f" 실패: {failed}" if failed else " 전부 기록됨."))


if __name__ == "__main__":
    main()