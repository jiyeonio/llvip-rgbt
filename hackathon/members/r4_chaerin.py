"""r4_chaerin.py — A안: 재학습 없는 탐색 (방법 A 라벨 R0 체크포인트 + 중복 억제 변형).

실행 전에 목록을 고정한다 (결과 보고 추가·삭제하지 않음). 기준은 Y2(conf×p → 선형 soft-NMS IoU≥0.5).
  Z1  3시드 확률 평균(앙상블) → conf×p̄ → 선형 NMS 0.5        앙상블이 시드 흔들림을 줄이는가
  Z2  conf×p^0.5 → 선형 NMS 0.5                                 ResNet 영향을 줄이면 더 나은가
  Z3  conf×p → 선형 NMS 임계 0.4   / Z4  임계 0.6               NMS 임계값에 얼마나 민감한가 (최적값 찾기 아님)
  Z5  conf×p → 순차 soft-NMS (깎인 점수를 다시 순서에 반영)       Y2 의 단순형과 차이가 있는가
  Z6  [B 라벨(1:1) R0-1to1 체크포인트] conf×p → 선형 NMS 0.5     라벨 A vs B 를 NMS 까지 같은 조건으로 비교 (Y2 의 짝)

    python hackathon/members/r4_chaerin.py smoke   # 점검(기록 안 됨): Y2 와 같은 설정이 Y2 숫자를 재현하는지 (약 4분)
    python hackathon/members/r4_chaerin.py all     # Z1~Z5 기록 (약 10분)
    python hackathon/members/r4_chaerin.py Z3      # 한 실험만
"""
import os
import sys
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r3_chaerin as M          # noqa: E402  (r3 의 확률 캐시·중복 억제·점수 만들기를 그대로 재사용)

OWNER = "채린"


def suppress_seq(df, mask, score, kind, t):
    """순차 soft-NMS. 점수가 가장 높은 박스부터 하나씩 확정하고, 확정한 박스와 겹친 아직 안 확정된 재점수 대상 박스의
    점수를 깎는다. 깎인 점수는 다음 순서에 반영된다 (여러 박스에 눌리면 곱해서 깎임)."""
    score = np.asarray(score, float)
    new = score.copy()
    boxes = df[["x1", "y1", "x2", "y2"]].to_numpy(float)
    mask = np.asarray(mask, bool)
    for _, idx in df.groupby("image_id").indices.items():
        idx = np.asarray(idx)
        m = mask[idx]
        if not m.any():
            continue
        s = score[idx].copy()
        iou = M._iou_matrix(boxes[idx])
        np.fill_diagonal(iou, 0.0)
        done = np.zeros(len(idx), bool)
        for _ in range(len(idx)):
            cur = np.where(done, -np.inf, s)
            k = int(np.argmax(cur))
            if not np.isfinite(cur[k]):
                break
            done[k] = True
            ov = iou[k]
            if kind == "lin":
                pen = np.where(ov >= t, 1.0 - ov, 1.0)
            elif kind == "gauss":
                pen = np.exp(-(ov ** 2) / t)
            else:
                raise ValueError(kind)
            dec = (~done) & m
            s[dec] = s[dec] * pen[dec]
        new[idx[m]] = s[m]
    return new


_PB = {}


def seed_probs_B(env, cands, seed):
    """B 라벨(1:1) R0-1to1 체크포인트의 B 확률 (r1_chaerin 의 체크포인트 폴더). 시드마다 한 번만 계산."""
    key = (seed, len(cands))
    if key not in _PB:
        import torch
        r = M.R()
        r.MODE = "r7a"                                   # ckpt_path 가 seed{n}.pt 를 가리키게 함
        path = r.ckpt_path(seed)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} 없음. r1_chaerin.py full 체크포인트가 필요합니다")
        model = r.build(pretrain=False).to(env.device)
        model.load_state_dict(torch.load(path, map_location=env.device))
        print(f"  [R0-1to1 시드 {seed}] 확률 계산", flush=True)
        _PB[key] = M._probs(env, cands, model)
    return _PB[key]


def rescore_Z(alpha=1.0, nms=("lin", 0.5), ens=False, seq=False, label="A"):
    def f(state, cands, env):
        conf = cands["conf"].to_numpy(float)
        if label == "B":
            mask, p = seed_probs_B(env, cands, state["seed"])
        elif ens:
            ps = [M.seed_probs_A(env, cands, s) for s in (0, 1, 2)]
            mask, p = ps[0][0], np.mean([x[1] for x in ps], axis=0)
        else:
            mask, p = M.seed_probs_A(env, cands, state["seed"])
        s = M.make_score(conf, mask, np.power(p, alpha), "comb")
        fn = suppress_seq if seq else M.suppress_by
        return fn(cands, mask, s, nms[0], nms[1])
    return f


S3 = (0, 1, 2)
BASE = "R0 시드의 방법 A 라벨(IoU≥0.5) ResNet, 재학습 없음"
DET = "결정적(학습·무작위 없음)이라 시드 1회, σ 없음"
EXPERIMENTS = [
    dict(id="Z1-A-ens3-nmsLin", change="앙상블 + 중복 억제", seeds=(0,), note="탐색(사전 고정 목록) · " + DET,
         setting=f"3시드 확률 평균 → conf × p̄ → 선형 soft-NMS(IoU≥0.5) · {BASE}", fn=rescore_Z(ens=True)),
    dict(id="Z2-A-a0.5-nmsLin", change="결합(α) + 중복 억제", seeds=S3, note="탐색(사전 고정 목록)",
         setting=f"conf × p^0.5 → 선형 soft-NMS(IoU≥0.5) · {BASE}", fn=rescore_Z(alpha=0.5)),
    dict(id="Z3-A-nmsLin0.4", change="NMS 임계 민감도", seeds=S3, note="탐색(사전 고정 목록) · 최적값 찾기 아님",
         setting=f"conf × p → 선형 soft-NMS(IoU≥0.4) · {BASE}", fn=rescore_Z(nms=("lin", 0.4))),
    dict(id="Z4-A-nmsLin0.6", change="NMS 임계 민감도", seeds=S3, note="탐색(사전 고정 목록) · 최적값 찾기 아님",
         setting=f"conf × p → 선형 soft-NMS(IoU≥0.6) · {BASE}", fn=rescore_Z(nms=("lin", 0.6))),
    dict(id="Z5-A-seqSoftNMS", change="순차 soft-NMS", seeds=S3, note="탐색(사전 고정 목록)",
         setting=f"conf × p → 순차 선형 soft-NMS(IoU≥0.5, 깎인 점수 재정렬) · {BASE}", fn=rescore_Z(seq=True)),
    dict(id="Z6-B-nmsLin", change="라벨 B + 중복 억제", seeds=S3, note="탐색(사전 고정 목록) · Y2 의 짝(라벨만 다름)",
         setting="1:1 라벨 ResNet(R0-1to1 시드 3개, 재학습 없음) · conf × p → 선형 soft-NMS(IoU≥0.5)",
         fn=rescore_Z(label="B")),
]


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    todo = EXPERIMENTS if mode == "all" else [e for e in EXPERIMENTS if mode and e["id"].startswith(mode)]
    if mode != "smoke" and not todo:
        sys.exit("사용: python r4_chaerin.py [smoke | all | Z1 | Z2 | Z3 | Z4 | Z5 | Z6]")
    lb = M.R().lb
    if mode == "smoke":
        print("점검: Y2 와 같은 설정(conf×p → 선형 NMS 0.5, 시드 3개)이 Y2 숫자를 재현하는지")
        print("      기대값: Q1 0.703 / 겹침 0.794 / 전체 0.893 | ΔQ1 +1.8%p")
        lb.submit("Z0-check", OWNER, "점검", "Y2 재현 점검", M.train_A, rescore_Z(), seeds=S3, dry_run=True)
        return
    failed = []
    for e in todo:
        print(f"\n######## {e['id']} ({e['change']}) ########", flush=True)
        try:
            lb.submit(e["id"], OWNER, e["change"], e["setting"], M.train_A, e["fn"], seeds=e["seeds"], note=e["note"])
        except Exception:
            traceback.print_exc()
            failed.append(e["id"])
    print("\n완료." + (f" 실패: {failed}" if failed else " 전부 기록됨."))


if __name__ == "__main__":
    main()