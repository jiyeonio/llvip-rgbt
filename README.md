# LLVIP 평가·분석 (2번 파트)

LLVIP RGB-열화상 야간 보행자 탐지에서, 탐지기 후보(CSV)를 받아 **작은 사람(Q1)·겹친 사람을 얼마나 놓치는지**,
**점수가 어느 쪽으로 치우치는지**, **그룹별 합격선으로 개선되는지**를 평가하는 저장소입니다.


## 폴더 구조

```
llvip_eval/
├─ code/                 코드 (아래 표)
├─ candidates/           후보 CSV: candidates_{모델}_{split}.csv
├─ data/
│  ├─ LLVIP/             원본 데이터 (Annotations/, visible/, infrared/)
│  ├─ fused_avg/         평균 융합 이미지 (fuse.py가 만듦)
│  ├─ yolo_proxy/        proxy 재학습용 YOLO 데이터셋
│  ├─ yolo_A/            A 학습용 YOLO 데이터셋 (train_A.py가 만듦)
│  └─ split.csv          A / B / test 분리 (1번 담당 제공)
├─ models/               학습 가중치 (proxy_retrain/, A_avg/ 등)
├─ results/E1, E2, E3/   실험 결과 (png, csv)
```

## 코드 파일

| 파일 | 하는 일 |
|---|---|
| `code/eval.py` | 공통 평가 함수. 정답 읽기(`load_gt`), 후보 읽기(`load_candidates`), 1:1 매칭(`match`), FPPI·Recall(`fppi_recall`), 미탐 원인(`miss_causes`), 임계값 스윕(`sweep`) |
| `code/fuse.py` | RGB와 열화상을 0.5:0.5로 픽셀 평균해 융합 이미지를 만듦 (`fuse_avg`) |
| `code/prepare_proxy_data.py` | proxy 재학습용 YOLO 데이터셋을 만듦 (Train `[::4]`, XML → YOLO 라벨) |
| `code/train_proxy.py` | proxy 재학습 (yolo11n.pt, imgsz 640, epochs 28, seed 0) |
| `code/train_A.py` | **1번 담당용.** split.csv 검사 → 평균 융합 → A로 학습(val은 A 안에서 segment 단위) → B·Test 후보 저장을 한 번에 |
| `code/predict.py` | 가중치로 후보 CSV를 만듦 (conf 0.001, iou 0.7, imgsz 640, max_det 1000). `predict_to_csv` 함수로도 씀 |
| `code/run_E1.py` | E1: FPPI-Recall 곡선, FPPI 0.25/0.5/1.0에서의 Recall, 미탐 원인 표 |
| `code/run_E2.py` | E2: 점수 치우침 (크기별·겹침별 reliability diagram) |
| `code/run_E3.py` | E3: 그룹별 합격선. **B 후보로만** 실행됨. B 전체로 고르기(in-sample, 참고용) + segment 단위 5-fold 교차검증 |
| `code/test_eval.py`, `code/test_E3.py` | 가짜 데이터 테스트 |

## LLVIP 받기

```bash
cd ~/llvip_eval/data
wget https://huggingface.co/datasets/UserNae3/LLVIP/resolve/main/LLVIP.zip
unzip -q LLVIP.zip            # 약 4GB, 풀면 data/LLVIP/ 가 생김
```

받은 뒤 **Test 정답이 8,302개인지** 확인합니다 (환경을 만든 뒤 실행).

```bash
cd ~/llvip_eval
.venv/bin/python -c "import sys; sys.path.insert(0,'code'); import eval as E; g=E.load_gt('test'); print(len(g.attrs['image_ids']), len(g))"
# 기대값: 3463 8302
```

- 이미지 쌍은 train 12,025, test 3,463입니다. visible과 infrared의 파일 이름이 같아야 합니다.
- `Annotations/`에는 train과 test XML이 한 폴더에 섞여 있습니다. split은 `visible/train`, `visible/test`의 파일 이름으로 구분합니다.
- 확인이 끝나면 `LLVIP.zip`은 지워도 됩니다.

## 환경 만들기

시스템 Python에는 pip 설치 모듈(`ensurepip`)이 없으므로 `uv`로 만듭니다.

```bash
cd ~/llvip_eval
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python \
    ultralytics numpy pandas matplotlib opencv-python-headless lxml koreanize-matplotlib
```

- `koreanize-matplotlib`는 그래프 한글 폰트(나눔고딕)용입니다.
- GPU가 있어야 학습과 예측이 현실적인 시간에 끝납니다 (RTX A5000에서 proxy 학습 약 7분).

## 1번 담당이 할 일 (순서대로)

### 1) split.csv 만들기

[split.csv 형식](#splitcsv-형식)대로 `data/split.csv`에 저장합니다.

### 2) 계획 확인 (아무것도 쓰지 않음)

```bash
.venv/bin/python code/train_A.py --plan
```

- split.csv 형식을 검사합니다:
  - split 값이 A / B / test인지, image_id가 6자리이고 중복이 없는지
  - 한 segment가 여러 split에 걸치지 않는지
  - A·B는 LLVIP train에, test는 LLVIP test에 있는지
- 문제가 있으면 무엇이 틀렸는지 출력하고 멈춥니다.
- 통과하면 A/B/test 이미지 수, val로 뗄 segment, 저장될 파일 이름을 보여 줍니다.

### 3) 평균 융합 → A로 학습 → B·Test 후보 저장 (한 번에)

```bash
.venv/bin/python code/train_A.py
```

| 단계 | 내용 | 출력 |
|---|---|---|
| 평균 융합 | A·B는 LLVIP train, test는 LLVIP test에서 RGB와 열화상을 0.5:0.5로 평균 | `data/fused_avg/{A,B,test}/` |
| 데이터셋 | XML → YOLO 라벨, person 1클래스 (이미지 밖 좌표는 잘라냄) | `data/yolo_A/` |
| val 분리 | **A 안에서 segment 단위로** 뗌. seed 0으로 segment를 섞어 A 이미지의 10% 이상이 될 때까지. B와 test는 학습에도 val에도 쓰지 않음 | `models/A_avg/A_val_segments.csv` |
| 학습 | proxy 재학습과 같은 설정: `yolo11n.pt`(COCO), imgsz 640, epochs 28, seed 0, deterministic, batch 16 | `models/A_avg/weights/best.pt` |
| 후보 저장 | conf 0.001, iou 0.7, imgsz 640, max_det 1000 | `candidates/candidates_avg_B.csv`<br>`candidates/DO_NOT_OPEN_candidates_avg_test.csv` |

- 마지막에 후보 파일별로 이미지 수, 후보 수, 이미지당 최대 후보 수가 출력됩니다. `⚠ max_det 1000에 닿은 이미지 있음`이 보이면 후보가 잘렸을 수 있으니 알려 주세요.
- 후보 파일이 이미 있으면 덮어쓰지 않고 멈춥니다. 덮어쓰려면 `--overwrite`를 붙입니다.
- 자주 쓰는 옵션:
  - `--name`: 저장 폴더 이름 (기본 `A_avg`)
  - `--model-tag`: 후보 파일 이름의 모델 부분 (기본 `avg`, 예: `4ch`)
  - `--val-frac`, `--val-seed`: val 비율과 seed
  - `--skip-train --weights <best.pt>`: 학습 없이 후보만 다시 저장
- **Test 후보(`DO_NOT_OPEN_…`)는 열어 보지 말고 커밋하지 마세요.**
  - 결정과 발표 수치는 B 후보로만 냅니다. 새 탐지기(avg, 4ch)의 Test 후보는 분석에 쓰지 않습니다.
  - 커밋을 막으려면 `.gitignore`에 아래 줄을 넣으세요.
    ```
    candidates/DO_NOT_OPEN_*
    **/DO_NOT_OPEN_*
    ```

## 파일 형식

### 후보 CSV 형식

파일 이름: `candidates/candidates_{모델}_{split}.csv`

- 모델: `proxy` / `avg` / `4ch`
- split: `B` / `test`
- 예: `candidates_avg_B.csv`

```
image_id,x1,y1,x2,y2,conf
010001,1023.377258,331.265564,1142.373291,570.388062,0.835560
```

| 열 | 설명 |
|---|---|
| `image_id` | **6자리 문자열** (`010001`처럼 앞의 0 유지). pandas로 읽을 때 `dtype={"image_id": str}` |
| `x1, y1, x2, y2` | 박스 좌상단·우하단, **원본 1280×1024 픽셀 좌표** (640으로 줄인 좌표가 아님) |
| `conf` | 점수. conf 0.001 이상 후보를 모두 저장 |

### split.csv 형식

경로: `data/split.csv`

```
image_id,segment_id,split
010001,0,A
010002,0,A
...
190001,57,test
```

| 열 | 설명 |
|---|---|
| `image_id` | **6자리 문자열**. A·B는 LLVIP train, test는 LLVIP test의 파일 이름 |
| `segment_id` | 연속 촬영 구간 번호. 같은 segment는 반드시 같은 split에 넣습니다. E3 교차검증의 fold도 segment 단위로 나눕니다. |
| `split` | 값은 정확히 **`A` / `B` / `test`** 중 하나 (대소문자 구분, 앞뒤 공백 없음) |

- A는 학습용, B는 분석과 결정용, test는 최종 확인용입니다.
- 코드는 `split == "B"`로 B 이미지를 고릅니다. `b`나 `B ` 같은 값은 0장으로 처리됩니다.

## 분석 실행 (2번 파트)

```bash
.venv/bin/python code/run_E1.py candidates/candidates_avg_B.csv
.venv/bin/python code/run_E2.py candidates/candidates_avg_B.csv
.venv/bin/python code/run_E3.py candidates/candidates_avg_B.csv   # --mode insample|cv|both (기본 both)
```

- 결과 파일 이름에는 후보 파일의 모델·split이 들어갑니다 (예: `results/E1/E1_curve_all_avg_B.png`).
- proxy(Test) 결과는 코드 검증용입니다. csv 첫 줄에 그렇게 표시되고, 발표 수치로 쓰지 않습니다.
- 테스트:
  ```bash
  .venv/bin/python code/test_eval.py
  .venv/bin/python code/test_E3.py
  ```
