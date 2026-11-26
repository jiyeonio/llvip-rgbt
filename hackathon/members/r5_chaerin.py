"""r5_chaerin.py — 학습 변형 탐색 (방법 A 라벨 ResNet 을 다시 학습하는 실험들).

통제 설계: 모든 실험은 **같은 평가 레시피**로 채점한다 = conf × p^0.5 → 선형 soft-NMS(IoU≥0.5)  (Z2 와 같은 후처리).
  → 실험 간 차이는 '학습 방식'에서만 나온다. 대조군 C0 는 R0 와 같은 설정을 이 코드로 다시 학습한 것.
  C0-ctrl   대조군 (R0 설정 그대로: 4채널·여백20%·112·증강없음·IoU≥0.5 라벨·3epoch)
  C0-tta    C0 체크포인트에 좌우반전 TTA (재학습 없음)
  C1-aug1   + 좌우반전 + 밝기/대비 변화(RGB·열화상 따로)
  C2-aug2   + aug1 + 박스 위치·크기 흔들기(±10%) + 랜덤 지우기(25%)   (장면 외우기·가려짐 대응)
  C3-margin50  여백 20% → 50% (주변 맥락)
  C4-size160   입력 112 → 160
  C5-softIoU   라벨을 0/1 이 아니라 IoU 값(≥0.5)으로: 잘 맞는 박스일수록 높은 점수 → NMS 에 유리
  C6-ep6cos    6 epoch + 코사인 학습률
  C7-irOnly    열화상 1채널만
  C8-r34       ResNet34 (사전학습 가중치 다운로드가 막히면 실패할 수 있음)

    python hackathon/members/r5_chaerin.py smoke     # 점검(기록 안 됨, 약 3분): 코드가 돌아가는지
    python hackathon/members/r5_chaerin.py all       # 전부 (약 3~4시간, screen 필수)
    python hackathon/members/r5_chaerin.py C1        # ID 가 C1 로 시작하는 것만
"""
import copy
import math
import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r3_chaerin as M          # noqa: E402  (R(), make_score, suppress_by, r0_dir 재사용)

OWNER = "채린"
ALPHA, NMS = 0.5, ("lin", 0.5)                    # 평가 레시피 (고정)
RECIPE = "평가: conf × p^0.5 → 선형 soft-NMS(IoU≥0.5) 고정"

BASE = dict(arch="resnet18", idx=[0, 1, 2, 3], margin=0.2, size=112, flip=False, color=False, jitter=0.0,
            erase=0.0, label="a", epochs=3, lr=3e-4, cosine=False, wd=0.01, batch=128, tta=False)


# ----------------------------------------------------------------------------- 순수 함수 (모델 불필요)
def _iou_cross(a, b):
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    ab = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    return inter / np.maximum(aa[:, None] + ab[None, :] - inter, 1e-9)


def max_iou_per_cand(cands, gts):
    """후보마다 같은 이미지 정답 중 최대 IoU (정답이 없으면 0)."""
    out = np.zeros(len(cands))
    cb = cands[["x1", "y1", "x2", "y2"]].to_numpy(float)
    gb = gts[["x1", "y1", "x2", "y2"]].to_numpy(float)
    gi = gts.groupby("image_id").indices
    for iid, ci in cands.groupby("image_id").indices.items():
        g = gi.get(iid)
        if g is None or len(g) == 0:
            continue
        out[ci] = _iou_cross(cb[ci], gb[g]).max(axis=1)
    return out


def make_labels(mi, kind):
    """kind='a': IoU≥0.5 이면 1 (R0 와 같음) / 'soft': IoU≥0.5 이면 IoU 값, 아니면 0"""
    pos = mi >= 0.5
    return np.where(pos, 1.0 if kind == "a" else mi, 0.0)


def jitter_box(x1, y1, x2, y2, shift, scale, rnd):
    w, h = x2 - x1, y2 - y1
    cx = (x1 + x2) / 2 + rnd(-shift, shift) * w
    cy = (y1 + y2) / 2 + rnd(-shift, shift) * h
    s = 1.0 + rnd(-scale, scale)
    return cx - w * s / 2, cy - h * s / 2, cx + w * s / 2, cy + h * s / 2


def crop_xy(vis, ir, box, margin, size):
    """r1/RC.crop 과 같은 방식(여백 비율 margin, size×size). 흔든 박스가 이미지 밖이어도 안 터지게 보호."""
    import cv2
    H, W = ir.shape
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    a1 = min(int(max(0, x1 - margin * w)), W - 1); b1 = min(int(max(0, y1 - margin * h)), H - 1)
    a2 = int(min(W, x2 + margin * w)); b2 = int(min(H, y2 + margin * h))
    a2, b2 = max(a2, a1 + 1), max(b2, b1 + 1)
    img = np.dstack([vis[b1:b2, a1:a2], ir[b1:b2, a1:a2]])
    return cv2.resize(img, (size, size))


# ----------------------------------------------------------------------------- 데이터셋·증강 (torch 필요)
def augment(x, c):
    import torch
    r = lambda a, b: a + (b - a) * float(torch.rand(1))
    if c["flip"] and r(0, 1) < 0.5:
        x = x.flip(-1)
    if c["color"]:
        x = x.clone()
        x[0:3] = x[0:3] * r(0.75, 1.25) + r(-0.1, 0.1)       # RGB
        x[3:4] = x[3:4] * r(0.75, 1.25) + r(-0.1, 0.1)       # 열화상
        x = x.clamp(0, 1)
    if c["erase"] > 0 and r(0, 1) < c["erase"]:
        _, H, W = x.shape
        eh, ew = max(1, int(H * r(0.15, 0.5))), max(1, int(W * r(0.15, 0.5)))
        y0, x0 = int((H - eh) * r(0, 1)), int((W - ew) * r(0, 1))
        x = x.clone()
        x[:, y0:y0 + eh, x0:x0 + ew] = r(0, 1)
    return x


def make_ds(cands, cfg, train, labels=None):
    import torch
    r = M.R()

    class DS(r.Crops):
        def __init__(self):
            super().__init__(cands, train=False, with_label=False)      # RC.CropDataset 초기화 재사용
            self.margin, self.size = cfg["margin"], cfg["size"]
            self.cfg, self.is_train, self.lab = cfg, train, labels

        def __getitem__(self, i):
            import cv2
            c, row, root = self.cfg, self.c.iloc[i], r.RC.ROOT
            vis = cv2.imread(f"{root}/visible/{self.subset}/{row.image_id}.jpg")[:, :, ::-1]
            ir = cv2.imread(f"{root}/infrared/{self.subset}/{row.image_id}.jpg", cv2.IMREAD_GRAYSCALE)
            box = (row.x1, row.y1, row.x2, row.y2)
            if self.is_train and c["jitter"] > 0:
                box = jitter_box(*box, c["jitter"], c["jitter"], lambda a, b: a + (b - a) * float(torch.rand(1)))
            x = torch.from_numpy(crop_xy(vis, ir, box, self.margin, self.size).astype(np.float32) / 255.).permute(2, 0, 1)
            if self.is_train:
                x = augment(x, c)
            x = ((x - self.MEAN) / self.STD)[c["idx"]]
            y = float(self.lab[i]) if self.lab is not None else 0.0
            return x, torch.tensor(y)

    return DS()


def ckpt_dir(tag):
    return os.path.join(os.path.dirname(M.r0_dir()), "R5", tag)


def build_model(cfg, env, pretrain):
    r = M.R()
    r.ARCH, r.IDX = cfg["arch"], cfg["idx"]
    return r.build(pretrain=pretrain).to(env.device)


def train_cfg(cfg, tag):
    def f(env, seed):
        import torch
        from torch.nn import BCEWithLogitsLoss
        from torch.utils.data import DataLoader
        r = M.R()
        if hasattr(env, "llvip_dir"):
            r.RC.ROOT = env.llvip_dir
        tr = env.train_cands()
        tr = tr.assign(_mi=max_iou_per_cand(tr, env.train_gts()))
        lo, hi = env.rescore_range
        tr = tr[(tr.conf >= lo) & (tr.conf <= hi)].reset_index(drop=True)
        y = make_labels(tr["_mi"].to_numpy(), cfg["label"])
        print(f"\n=== [{tag}] 시드 {seed} 학습 시작 · 후보 {len(tr):,}개 · 양성(라벨>0) {100 * (y > 0).mean():.1f}% ===", flush=True)
        dl = DataLoader(make_ds(tr, cfg, True, y), batch_size=cfg["batch"], shuffle=True, num_workers=4,
                        generator=torch.Generator().manual_seed(seed))
        model = build_model(cfg, env, True)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
        nb = len(dl)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"] * nb) if cfg["cosine"] else None
        lossf = BCEWithLogitsLoss()
        model.train()
        t0 = time.time()
        for ep in range(cfg["epochs"]):
            te, tot, n = time.time(), 0.0, 0
            for b, (x, yy) in enumerate(dl, 1):
                x, yy = x.to(env.device), yy.to(env.device).float()
                loss = lossf(model(x).squeeze(1), yy)
                opt.zero_grad(); loss.backward(); opt.step()
                if sched:
                    sched.step()
                tot, n = tot + loss.item() * len(yy), n + len(yy)
                if b % 100 == 0 or b == nb:
                    print(f"    [{ep + 1}/{cfg['epochs']}] 배치 {b}/{nb}  loss {tot / n:.4f}  경과 {(time.time() - te) / 60:.1f}분", flush=True)
            print(f"  epoch {ep + 1}/{cfg['epochs']}  loss {tot / n:.4f} · 누적 {(time.time() - t0) / 60:.1f}분", flush=True)
        os.makedirs(ckpt_dir(tag), exist_ok=True)
        torch.save(model.state_dict(), os.path.join(ckpt_dir(tag), f"seed{seed}.pt"))
        return {"model": model, "seed": seed}
    return f


def load_cfg(cfg, src_tag):
    """재학습 없이 다른 실험(src_tag)의 체크포인트를 불러온다."""
    def f(env, seed):
        import torch
        path = os.path.join(ckpt_dir(src_tag), f"seed{seed}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} 없음. {src_tag} 를 먼저 돌리세요")
        model = build_model(cfg, env, False)
        model.load_state_dict(torch.load(path, map_location=env.device))
        return {"model": model, "seed": seed}
    return f


def rescore_cfg(cfg):
    def f(state, cands, env):
        import torch
        from torch.utils.data import DataLoader
        r = M.R()
        r.ARCH, r.IDX = cfg["arch"], cfg["idx"]
        if hasattr(env, "llvip_dir"):
            r.RC.ROOT = env.llvip_dir
        mask = np.asarray(env.rescore_mask(cands), bool)
        dl = DataLoader(make_ds(cands[mask].reset_index(drop=True), cfg, False), batch_size=512, shuffle=False, num_workers=4)
        model, ps = state["model"].eval(), []
        with torch.no_grad():
            for x, _ in dl:
                x = x.to(env.device)
                p = torch.sigmoid(model(x).squeeze(1))
                if cfg["tta"]:
                    p = (p + torch.sigmoid(model(x.flip(-1)).squeeze(1))) / 2
                ps.append(p.float().cpu().numpy())
        conf = cands["conf"].to_numpy(float)
        s = M.make_score(conf, mask, np.power(np.concatenate(ps), ALPHA), "comb")
        return M.suppress_by(cands, mask, s, *NMS)
    return f


# ----------------------------------------------------------------------------- 실험 목록
def cfg_of(**kw):
    c = copy.deepcopy(BASE)
    c.update(kw)
    return c


S3 = (0, 1, 2)
NOTE = "탐색(학습 변형, 대조군 C0 와 비교)"
E_ = []


def add(id_, change, setting, cfg, train=None, seeds=S3):
    E_.append(dict(id=id_, change=change, setting=f"{setting} · {RECIPE}", cfg=cfg, seeds=seeds,
                   train=train or train_cfg(cfg, id_)))


C0 = cfg_of()
add("C0-ctrl", "대조군(R0 설정 재학습)", "ResNet18·4채널·여백20%·112·증강없음·IoU≥0.5 라벨·3epoch·재학습", C0)
add("C0-tta", "TTA", "C0 체크포인트에 좌우반전 평균(재학습 없음)", cfg_of(tta=True), train=load_cfg(cfg_of(tta=True), "C0-ctrl"))
add("C1-aug1", "증강", "C0 + 좌우반전 + 밝기/대비", cfg_of(flip=True, color=True))
add("C2-aug2", "증강(강)", "C1 + 박스 흔들기(±10%) + 랜덤 지우기(25%)", cfg_of(flip=True, color=True, jitter=0.1, erase=0.25))
add("C3-margin50", "입력(여백)", "C0 + 여백 50%", cfg_of(margin=0.5))
add("C4-size160", "입력(크기)", "C0 + 입력 160×160", cfg_of(size=160))
add("C5-softIoU", "라벨(soft IoU)", "C0 + 라벨을 IoU 값(≥0.5)으로", cfg_of(label="soft"))
add("C6-ep6cos", "학습(epoch)", "C0 + 6epoch + 코사인 학습률", cfg_of(epochs=6, cosine=True))
add("C7-irOnly", "입력(열화상만)", "C0 + 열화상 1채널만", cfg_of(idx=[3]))
add("C8-r34", "모델(ResNet34)", "C0 + ResNet34", cfg_of(arch="resnet34"))


def run(e, lb, dry_run=False, seeds=None):
    print(f"\n######## {e['id']} ({e['change']}) ########", flush=True)
    lb.submit(e["id"], OWNER, e["change"], e["setting"], e["train"], rescore_cfg(e["cfg"]),
              seeds=seeds or e["seeds"], note=NOTE, dry_run=dry_run)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    todo = E_ if mode == "all" else [e for e in E_ if mode and e["id"].startswith(mode)]
    if mode != "smoke" and not todo:
        sys.exit("사용: python r5_chaerin.py [smoke | all | C0 | C1 | ... | C8]")
    lb = M.R().lb
    if mode == "smoke":
        print("점검: C0 설정 1 epoch, 시드 1개, 기록 안 됨. 로그에 '양성(라벨>0) 23.8%' 가 보여야 함")
        c = cfg_of(epochs=1)
        e = dict(id="C-smoke", change="점검", setting="점검", cfg=c, seeds=(0,), train=train_cfg(c, "smoke"))
        run(e, lb, dry_run=True)
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