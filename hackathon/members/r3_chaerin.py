"""r3_chaerin.py — 2번 묶음.

 [A] 노션 계획대로 '방법 A 라벨(IoU≥0.5) ResNet + 중복 억제' (재학습 없음, results/R0/seed{0,1,2}.pt 사용)
     Y1  conf × p                     (A라벨 확률과 YOLO 점수 결합)
     Y2  conf × p  → soft-NMS 선형    (새 점수 기준으로 중복을 깎음)
     Y3  conf × p  → soft-NMS 가우시안
     Y4  p 로 교체 → soft-NMS 선형     (친구가 말한 방식 그대로: ResNet 확률 + NMS)
     Y5  p 로 교체 → soft-NMS 가우시안
 [B] 노션 R2 입력 어블레이션 (재학습 있음, 1:1 라벨, 결합은 conf × p)
     R2-rgb  RGB 3채널만 / R2-ir  열화상 1채널만   (4채널은 이미 R7-a-1to1)

    python hackathon/members/r3_chaerin.py smoke    # 점검(기록 안 됨)
    python hackathon/members/r3_chaerin.py A        # Y1~Y5 기록 (약 10분)
    python hackathon/members/r3_chaerin.py R2       # R2-rgb, R2-ir 기록 (약 40분, screen 권장)
    python hackathon/members/r3_chaerin.py Y3       # 한 실험만 (ID 앞부분)
"""
import os
import sys
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OWNER = "채린"
_R = None


def R():
    global _R
    if _R is None:
        import r1_chaerin
        _R = r1_chaerin
    return _R


# ----------------------------------------------------------------------------- 순수 함수
def _iou_matrix(b):
    x1 = np.maximum(b[:, None, 0], b[None, :, 0])
    y1 = np.maximum(b[:, None, 1], b[None, :, 1])
    x2 = np.minimum(b[:, None, 2], b[None, :, 2])
    y2 = np.minimum(b[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    return inter / np.maximum(area[:, None] + area[None, :] - inter, 1e-9)


def suppress_by(df, mask, score, kind, t):
    """새 점수(score) 기준 중복 억제. 재점수 대상(mask) 박스만, 같은 이미지에서 자기보다 score 가 높은 박스와의
    최대 IoU(mi) 로 점수를 깎는다. 나머지 박스는 score 그대로.
      lin: mi>=t 이면 ×(1-mi) / gauss: ×exp(-mi²/t).   (깎인 점수를 다시 순서에 반영하지 않는 단순형)"""
    score = np.asarray(score, float)
    new = score.copy()
    boxes = df[["x1", "y1", "x2", "y2"]].to_numpy(float)
    mask = np.asarray(mask, bool)
    for _, idx in df.groupby("image_id").indices.items():
        idx = np.asarray(idx)
        m = mask[idx]
        if not m.any():
            continue
        s = score[idx]
        iou = _iou_matrix(boxes[idx])
        np.fill_diagonal(iou, 0.0)
        pos = np.empty(len(idx), int)
        pos[np.argsort(-s, kind="stable")] = np.arange(len(idx))
        mi = np.where(pos[None, :] < pos[:, None], iou, 0.0).max(axis=1)
        if kind == "lin":
            pen = np.where(mi >= t, 1.0 - mi, 1.0)
        elif kind == "gauss":
            pen = np.exp(-(mi ** 2) / t)
        else:
            raise ValueError(kind)
        new[idx[m]] = s[m] * pen[m]
    return new


def make_score(conf, mask, p, how):
    """how='comb' → conf×p, 'rep' → p 로 교체 (재점수 대상만)."""
    new = np.array(conf, float, copy=True)
    new[mask] = new[mask] * p if how == "comb" else p
    return new


# ----------------------------------------------------------------------------- 확률 계산
def _probs(env, cands, model):
    import torch
    from torch.utils.data import DataLoader
    r = R()
    if hasattr(env, "llvip_dir"):
        r.RC.ROOT = env.llvip_dir
    model.eval()
    mask = np.asarray(env.rescore_mask(cands), bool)
    dl = DataLoader(r.Crops(cands[mask].reset_index(drop=True)), batch_size=512, shuffle=False, num_workers=4)
    with torch.no_grad():
        p = np.concatenate([torch.sigmoid(model(x.to(env.device)).squeeze(1)).float().cpu().numpy() for x, _ in dl])
    return mask, p


def r0_dir():
    """방법 A 로 학습한 R0 체크포인트 폴더 (r1 은 이 이름 뒤에 _1to1 을 붙여 둠)."""
    d = str(R().CKPT_DIR)
    return d[:-len("_1to1")] if d.endswith("_1to1") else d


_PA = {}


def seed_probs_A(env, cands, seed):
    """R0(방법 A 라벨) 체크포인트의 B 확률. 시드마다 한 번만 계산."""
    key = (seed, len(cands))
    if key not in _PA:
        import torch
        r = R()
        path = os.path.join(r0_dir(), f"seed{seed}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} 없음. R0 full 체크포인트가 필요합니다")
        model = r.build(pretrain=False).to(env.device)
        model.load_state_dict(torch.load(path, map_location=env.device))
        print(f"  [R0 시드 {seed}] 확률 계산", flush=True)
        _PA[key] = _probs(env, cands, model)
    return _PA[key]


def train_A(env, seed):
    return {"seed": seed}


def rescore_A(how, nms=None):
    def f(state, cands, env):
        conf = cands["conf"].to_numpy(float)
        mask, p = seed_probs_A(env, cands, state["seed"])
        s = make_score(conf, mask, p, how)
        if nms:
            s = suppress_by(cands, mask, s, nms[0], nms[1])
        return s
    return f


# ----------------------------------------------------------------------------- R2 입력 어블레이션
def train_ablation(channels, idx, tag):
    def f(env, seed):
        r = R()
        r.IDX = idx                                    # 입력 채널 선택 (RGB=[0,1,2], 열화상=[3])
        r.CKPT_DIR = r0_dir() + f"_1to1_{tag}"         # 다른 체크포인트를 덮어쓰지 않게
        r.MODE = "full"
        print(f"  [R2] 입력 채널: {channels}", flush=True)
        return r.train(env, seed)
    return f


def rescore_ablation(state, cands, env):
    conf = cands["conf"].to_numpy(float)
    mask, p = _probs(env, cands, state["model"])
    return make_score(conf, mask, p, "comb")


# ----------------------------------------------------------------------------- 실험 목록
S3 = (0, 1, 2)
A_SET = ("R0 시드 3개의 방법 A 라벨(IoU≥0.5) ResNet, 재학습 없음")
EXPERIMENTS = [
    dict(id="Y1-A-comb", change="점수 결합 (A라벨)", seeds=S3, train=train_A, rescore=rescore_A("comb"),
         setting=f"YOLO conf × p · {A_SET}"),
    dict(id="Y2-A-comb-nmsLin", change="결합 + 중복 억제", seeds=S3, train=train_A, rescore=rescore_A("comb", ("lin", 0.5)),
         setting=f"conf × p → 새 점수 기준 선형 soft-NMS(IoU≥0.5) · {A_SET}"),
    dict(id="Y3-A-comb-nmsGauss", change="결합 + 중복 억제", seeds=S3, train=train_A,
         rescore=rescore_A("comb", ("gauss", 0.5)), setting=f"conf × p → 가우시안 soft-NMS(σ=0.5) · {A_SET}"),
    dict(id="Y4-A-rep-nmsLin", change="확률 교체 + 중복 억제", seeds=S3, train=train_A,
         rescore=rescore_A("rep", ("lin", 0.5)), setting=f"p 로 교체 → 선형 soft-NMS(IoU≥0.5) · {A_SET}"),
    dict(id="Y5-A-rep-nmsGauss", change="확률 교체 + 중복 억제", seeds=S3, train=train_A,
         rescore=rescore_A("rep", ("gauss", 0.5)), setting=f"p 로 교체 → 가우시안 soft-NMS(σ=0.5) · {A_SET}"),
    dict(id="R2-rgb", change="입력 어블레이션", seeds=S3, train=train_ablation("RGB 3채널", [0, 1, 2], "rgb"),
         rescore=rescore_ablation, setting="ResNet18·RGB 3채널만·1:1 라벨·YOLO conf × p (재학습)"),
    dict(id="R2-ir", change="입력 어블레이션", seeds=S3, train=train_ablation("열화상 1채널", [3], "ir"),
         rescore=rescore_ablation, setting="ResNet18·열화상 1채널만·1:1 라벨·YOLO conf × p (재학습)"),
]
GROUPS = {"A": [e for e in EXPERIMENTS if e["id"].startswith("Y")],
          "R2": [e for e in EXPERIMENTS if e["id"].startswith("R2")]}


def run(e, lb, dry_run=False):
    print(f"\n######## {e['id']} ({e['change']}) ########", flush=True)
    lb.submit(e["id"], OWNER, e["change"], e["setting"], e["train"], e["rescore"], seeds=e["seeds"],
              note="탐색(결과 확인 후 추가한 실험)" if e["id"].startswith("Y") else "노션 R2", dry_run=dry_run)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    todo = GROUPS.get(mode) or [e for e in EXPERIMENTS if mode and e["id"].startswith(mode)]
    if mode != "smoke" and not todo:
        sys.exit("사용: python r3_chaerin.py [smoke | A | R2 | Y1..Y5 | R2-rgb | R2-ir]")
    lb = R().lb
    if mode == "smoke":
        print("점검: R0 시드0 확률로 '교체'하면 처음 R0 시드0 과 같아야 함 (Q1 0.254 / 겹침 0.508 / 전체 0.543)")
        lb.submit("Y0-check", OWNER, "점검", "R0 시드0 재현", train_A, rescore_A("rep"), seeds=(0,), dry_run=True)
        print("\n점검: 새 점수 기준 중복 억제가 돌아가고 숫자가 나오는지")
        lb.submit("Y0-check2", OWNER, "점검", "교체+억제 점검", train_A, rescore_A("rep", ("lin", 0.5)),
                  seeds=(0,), dry_run=True)
        return
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