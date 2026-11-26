import os, sys, math, time
import numpy as np, torch, torch.nn as nn, torchvision
from torch.utils.data import DataLoader
sys.path.insert(0, "/workspace/llvip_eval/hackathon")
import lb
import resnet_crop as RC
RC.ROOT = lb.Env.llvip_dir

CHANNELS = "rgbt"      # R1: "rgbt" / "rgb" / "t"
MARGIN   = 0.2         # R2
SIZE     = 112         # R3
PRETRAIN = True        # R4-a
ARCH     = "resnet18"  # R10
FLIP     = False       # R5-a
EPOCHS, LR, BATCH = 3, 3e-4, 128
IDX = {"rgbt": [0, 1, 2, 3], "rgb": [0, 1, 2], "t": [3]}[CHANNELS]
CKPT_DIR = "/workspace/llvip_eval/results/R0"
MODE = "quick"         # 맨 아래 main 에서 quick / full / r7a 로 바뀜


def ckpt_path(seed):
    return os.path.join(CKPT_DIR, f"{'quick_seed' if MODE == 'quick' else 'seed'}{seed}.pt")


class Crops(RC.CropDataset):
    def __init__(self, cands, train=False, with_label=False):
        super().__init__(cands, subset="train", size=SIZE, margin=MARGIN)
        self.train_, self.with_label = train, with_label

    def __getitem__(self, i):
        x = torch.from_numpy(self.crop(i).astype(np.float32) / 255.).permute(2, 0, 1)
        x = ((x - self.MEAN) / self.STD)[IDX]
        if FLIP and self.train_ and np.random.rand() < 0.5:
            x = x.flip(-1)
        y = self.label(i) if self.with_label else 0.0
        return x, torch.tensor(y)


def build(pretrain=True):
    m = getattr(torchvision.models, ARCH)(weights="DEFAULT" if pretrain else None)
    old = m.conv1
    new = nn.Conv2d(len(IDX), 64, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        for j, c in enumerate(IDX):
            new.weight[:, j] = old.weight[:, c] if c < 3 else old.weight.mean(dim=1)
    m.conv1 = new
    m.fc = nn.Linear(m.fc.in_features, 1)
    return m


def train(env, seed):
    if MODE == "r7a":                                   # 재학습 없음: R0 체크포인트를 불러옴
        path = ckpt_path(seed)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} 없음. 먼저 full 모드로 R0를 돌려 체크포인트를 만드세요")
        model = build(pretrain=False).to(env.device)
        model.load_state_dict(torch.load(path, map_location=env.device))
        return {"model": model}
    tr = env.train_cands()
    lo, hi = env.rescore_range
    tr = tr[(tr.conf >= lo) & (tr.conf <= hi)].reset_index(drop=True)
    print(f"\n=== 시드 {seed} 학습 시작 · 후보 {len(tr):,}개 ===", flush=True)
    dl = DataLoader(Crops(tr, train=True, with_label=True), batch_size=BATCH, shuffle=True, num_workers=4,
                    generator=torch.Generator().manual_seed(seed))
    model = build(PRETRAIN).to(env.device)
    opt, lossf = torch.optim.AdamW(model.parameters(), lr=LR), nn.BCEWithLogitsLoss()
    model.train()
    t_start, n_batches = time.time(), len(dl)
    for ep in range(EPOCHS):
        t_ep = time.time()
        tot = n = pos = 0
        for b, (x, y) in enumerate(dl, 1):
            x, y = x.to(env.device), y.to(env.device).float()
            loss = lossf(model(x).squeeze(1), y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot, n, pos = tot + loss.item() * len(y), n + len(y), pos + y.sum().item()
            if b % 50 == 0 or b == n_batches:
                el = time.time() - t_ep
                print(f"    [{ep + 1}/{EPOCHS}] 배치 {b}/{n_batches}  loss {tot / n:.4f}  경과 {el / 60:.1f}분  남은 {el / b * (n_batches - b) / 60:.1f}분", flush=True)
        pr = pos / n
        const = -(pr * math.log(pr) + (1 - pr) * math.log(1 - pr))
        print(f"  epoch {ep + 1}/{EPOCHS}  loss {tot / n:.4f}  (아무것도 안 배운 모델 {const:.4f}, 양성 {100 * pr:.1f}%) · 이번 epoch {(time.time() - t_ep) / 60:.1f}분 · 누적 {(time.time() - t_start) / 60:.1f}분", flush=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.save(model.state_dict(), ckpt_path(seed))
    return {"model": model}


@torch.no_grad()
def rescore(state, cands, env):
    model = state["model"].eval()
    mask = env.rescore_mask(cands)
    print(f"B 재점수 시작 · {int(mask.sum()):,}개 후보 (시간이 조금 걸림)", flush=True)
    dl = DataLoader(Crops(cands[mask].reset_index(drop=True)), batch_size=512, shuffle=False, num_workers=4)
    p = np.concatenate([torch.sigmoid(model(x.to(env.device)).squeeze(1)).float().cpu().numpy() for x, _ in dl])
    new = cands["conf"].to_numpy().copy()
    new[mask] = new[mask] * p if MODE == "r7a" else p      # R7-a: conf × ResNet 확률 / R0: ResNet 확률만
    return new


if __name__ == "__main__":
    MODE = sys.argv[1] if len(sys.argv) > 1 else "quick"
    if MODE not in ("quick", "full", "r7a"):
        sys.exit("사용: python r0_chaerin.py [quick | full | r7a]")
    if MODE == "quick":
        EPOCHS = 1
    R0 = "ResNet18·ImageNet 사전학습·4채널·여백 20%·112×112·증강 없음·BCE"
    id_, change, setting, seeds, dry = {
        "quick": ("R0", "기본형", R0, (0,), True),
        "full":  ("R0", "기본형", R0, (0, 1, 2), False),
        "r7a":   ("R7-a", "점수 결합", "YOLO conf × ResNet 확률 (R0 시드 3개 각각에 적용, 재학습 없음)", (0, 1, 2), False),
    }[MODE]
    lb.submit(id_, "채린", change, setting, train, rescore, seeds=seeds, dry_run=dry)
