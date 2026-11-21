"""E4-CNN 준비 코드: crop Dataset + 4채널 ResNet18 (1번 작성, 11/21)"""
import numpy as np, pandas as pd, cv2, torch, torch.nn as nn, torchvision, xml.etree.ElementTree as ET
from torch.utils.data import Dataset

ROOT = '/content/LLVIP'


def load_gt(iid):
    """정답 박스 (N,4) = xmin, ymin, xmax, ymax"""
    r = ET.parse(f'{ROOT}/Annotations/{iid}.xml').getroot()
    b = [[float(o.find(f'bndbox/{k}').text) for k in ('xmin','ymin','xmax','ymax')] for o in r.iter('object')]
    return np.array(b, dtype=np.float32).reshape(-1, 4)


def max_iou(box, gts):
    """box 하나와 정답 박스들 사이 IoU 중 최댓값"""
    if len(gts) == 0: return 0.0
    ix1 = np.maximum(box[0], gts[:,0]); iy1 = np.maximum(box[1], gts[:,1])
    ix2 = np.minimum(box[2], gts[:,2]); iy2 = np.minimum(box[3], gts[:,3])
    inter = np.clip(ix2-ix1, 0, None) * np.clip(iy2-iy1, 0, None)
    a = (box[2]-box[0])*(box[3]-box[1]); b = (gts[:,2]-gts[:,0])*(gts[:,3]-gts[:,1])
    return float((inter / np.maximum(a + b - inter, 1e-9)).max())


class CropDataset(Dataset):
    """후보 박스 + 여백을 잘라 RGB 3채널 + 열화상 1채널 = 4채널 텐서로 반환. 라벨 = 정답과 IoU>=0.5"""
    MEAN = torch.tensor([0.485, 0.456, 0.406, 0.449]).view(4,1,1)
    STD  = torch.tensor([0.229, 0.224, 0.225, 0.226]).view(4,1,1)

    def __init__(self, cands, subset='train', size=112, margin=0.2):
        self.c = cands.reset_index(drop=True)
        self.subset, self.size, self.margin = subset, size, margin
        self._gt = {}

    def __len__(self):
        return len(self.c)

    def crop(self, i):
        r = self.c.iloc[i]
        vis = cv2.imread(f'{ROOT}/visible/{self.subset}/{r.image_id}.jpg')[:, :, ::-1]
        ir  = cv2.imread(f'{ROOT}/infrared/{self.subset}/{r.image_id}.jpg', cv2.IMREAD_GRAYSCALE)
        H, W = ir.shape
        w, h = r.x2 - r.x1, r.y2 - r.y1
        x1 = int(max(0, r.x1 - self.margin*w)); y1 = int(max(0, r.y1 - self.margin*h))
        x2 = int(min(W, r.x2 + self.margin*w)); y2 = int(min(H, r.y2 + self.margin*h))
        x2, y2 = max(x2, x1+1), max(y2, y1+1)
        img = np.dstack([vis[y1:y2, x1:x2], ir[y1:y2, x1:x2]])
        return cv2.resize(img, (self.size, self.size))

    def label(self, i):
        r = self.c.iloc[i]
        if r.image_id not in self._gt:
            self._gt[r.image_id] = load_gt(r.image_id)
        return float(max_iou(np.array([r.x1, r.y1, r.x2, r.y2]), self._gt[r.image_id]) >= 0.5)

    def __getitem__(self, i):
        x = torch.from_numpy(self.crop(i).astype(np.float32) / 255.).permute(2, 0, 1)
        return (x - self.MEAN) / self.STD, torch.tensor(self.label(i))


def make_resnet18_4ch():
    """ImageNet 사전학습 ResNet18 → 입력 4채널(RGB+열화상), 출력 1개(사람일 확률의 logit)"""
    m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    old = m.conv1
    new = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        new.weight[:, :3] = old.weight
        new.weight[:, 3:] = old.weight.mean(dim=1, keepdim=True)
    m.conv1 = new
    m.fc = nn.Linear(m.fc.in_features, 1)
    return m
