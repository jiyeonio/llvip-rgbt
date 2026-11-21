# LLVIP RGB-T 야간 보행자 탐지: 평가·분석 저장소

LLVIP RGB-열화상 야간 보행자 탐지에서, 탐지기가 낸 후보(CSV)를 받아

- **작은 사람(Q1)·겹친 사람을 얼마나 놓치는지** (E1)
- **점수가 어느 쪽으로 치우치는지** (E2)
- **그룹별 합격선이나 재점수화로 개선되는지** (E3, E4)

를 평가하는 저장소입니다. 다음 단계인 ResNet 재판별기 준비 코드도 함께 있습니다.

## 폴더 구조

```
llvip_eval/
├─ code/                  코드 (아래 표)
├─ candidates/            후보 CSV: candidates_{모델}_{split}.csv
│  ├─ candidates_proxy_test.csv   proxy 재학습 탐지기의 Test 후보 (코드 검증용)
│  └─ candidates_avg_B.csv        A로 학습한 탐지기의 B 후보 (분석용)
├─ data/
│  ├─ split.csv           A / B / test 분리 (저장소에 포함)
│  ├─ LLVIP/              원본 데이터 (Annotations/, visible/, infrared/)   ← 저장소에 없음
│  ├─ fused_avg/          평균 융합 이미지 (fuse.py가 만듦)                 ← 저장소에 없음
│  ├─ yolo_proxy/, yolo_A/   YOLO 학습용 데이터셋                          ← 저장소에 없음
│  └─ features_*.parquet  E4 특징 캐시                                      ← 저장소에 없음
├─ weights/
│  └─ A_avg_best.pt       A로 학습한 YOLO11n 탐지기 (ResNet 학습 데이터 만들 때 사용)
├─ models/                학습 중간 산출물                                  ← 저장소에 없음
└─ results/E1, E2, E3, E4/   실험 결과 (png, csv)
```

## 코드 파일

| 파일 | 하는 일 |
| --- | --- |
| `code/eval.py` | 공통 평가 함수. 정답 읽기(`load_gt`), 후보 읽기(`load_candidates`), 1:1 매칭(`match`), FPPI·Recall(`fppi_recall`), 미탐 원인(`miss_causes`), 임계값 스윕(`sweep`), E4 특징(`geom_features`, `ir_features`, `rgb_features`, `extract_features`, `build_e42`) |
| `code/fuse.py` | RGB와 열화상을 0.5:0.5로 픽셀 평균해 융합 이미지를 만듦 (`fuse_avg`) |
| `code/prepare_proxy_data.py` | proxy 재학습용 YOLO 데이터셋을 만듦 (Train `[::4]`, XML → YOLO 라벨) |
| `code/train_proxy.py` | proxy 재학습 (yolo11n.pt, imgsz 640, epochs 28, seed 0) |
| `code/train_A.py` | split.csv 검사 → 평균 융합 → A로 학습(val은 A 안에서 segment 단위) → B·Test 후보 저장을 한 번에 |
| `code/predict.py` | 가중치로 후보 CSV를 만듦 (conf 0.001, iou 0.7, imgsz 640, max_det 1000). `predict_to_csv` 함수로도 씀 |
| `code/run_E1.py` | E1: FPPI-Recall 곡선, FPPI 0.25/0.5/1.0에서의 Recall, 미탐 원인 표 |
| `code/run_E2.py` | E2: 점수 치우침 (크기별·겹침별 reliability diagram) |
| `code/run_E3.py` | E3: 그룹별 합격선. B 후보로만 실행됨. B 전체로 고르기(in-sample, 참고용) + segment 단위 5-fold 교차검증 |
| `code/run_E4.py` | E4-0: 개선 여지 상한 / E4-1: LightGBM 재점수화 (특징 묶음 4개 + 로지스틱 회귀 비교 + E3 합격선 결합). B로만 실행 |
| `code/run_E4_2.py` | E4-2: 장면 의존 특징(위치·절대 밝기)을 뺀 재점수화. 로지스틱 회귀(메인), 제약 LightGBM, 진단 모델(y중심만 제거), 장면별 결과 |
| `code/plot_slides.py` | 발표용 그래프 2장 (미탐 원인 누적 막대, 개선 여지 상한 막대). 기존 결과 csv를 읽기만 함 |
| `code/resnet_crop.py` | ResNet 재판별기 준비 코드: 후보 crop Dataset(RGB+열화상 4채널, 112×112) + 4채널 ResNet18 (1번 작성) |
| `code/test_eval.py`, `code/test_E3.py`, `code/test_E4.py` | 가짜 데이터 테스트 |

## LLVIP 받기

```
cd ~/llvip_eval/data
wget https://huggingface.co/datasets/UserNae3/LLVIP/resolve/main/LLVIP.zip
unzip -q LLVIP.zip            # 약 4GB, 풀면 data/LLVIP/ 가 생김
```

받은 뒤 Test 정답이 8,302개인지 확인합니다 (환경을 만든 뒤 실행).

```
cd ~/llvip_eval
.venv/bin/python -c "import sys; sys.path.insert(0,'code'); import eval as E; g=E.load_gt('test'); print(len(g.attrs['image_ids']), len(g))"
# 기대값: 3463 8302
```

- 이미지 쌍은 train 12,025, test 3,463입니다. visible과 infrared의 파일 이름이 같아야 합니다.
- `Annotations/`에는 train과 test XML이 한 폴더에 섞여 있습니다. split은 `visible/train`, `visible/test`의 파일 이름으로 구분합니다.
- 확인이 끝나면 LLVIP.zip은 지워도 됩니다.

## 환경 만들기

시스템 Python에는 pip 설치 모듈(ensurepip)이 없으므로 uv로 만듭니다.

```
cd ~/llvip_eval
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python \
    ultralytics numpy pandas matplotlib opencv-python-headless lxml koreanize-matplotlib \
    lightgbm scikit-learn pyarrow
```

- `koreanize-matplotlib`: 그래프 한글 폰트(나눔고딕)
- `lightgbm`, `scikit-learn`, `pyarrow`: E4 재점수화와 특징 캐시(parquet)
- `torch`, `torchvision`은 ultralytics를 설치할 때 같이 들어옵니다 (ResNet 코드에서 사용)
- GPU가 있어야 학습과 예측이 현실적인 시간에 끝납니다 (RTX A5000에서 proxy 학습 약 7분)

## 데이터 분리와 탐지기 (1번 담당, 완료)

### 현재 split (data/split.csv)

- segment = 파일 이름 앞 두 자리 (같은 번호 = 같은 장소). Train 01~18, 25 / Test 19~24, 26
- 장면 단위로 섞어 B가 약 30%가 되게 나눔. seed 42는 작은 사람 비율이 A 21.8% / B 15.4%로 몰려서 규칙대로 seed 43 사용

| split | 이미지 | 장면 | 정답 | 작은 사람 비율 |
| --- | --- | --- | --- | --- |
| A (탐지기 학습) | 8,302 | 13개 (01, 02, 05, 06, 08, 10, 11, 12, 14, 15, 17, 18, 25) | 22,994 | 19.5% |
| B (분석·결정) | 3,723 | 6개 (03, 04, 07, 09, 13, 16) | 11,138 | 20.6% |
| test (최종 확인) | 3,463 | 7개 (19~24, 26) | 8,302 | – |

### 탐지기 (weights/A_avg_best.pt)

YOLO11n(COCO 사전학습), 평균 융합 입력, A만으로 학습(A 안의 장면 2개 940장을 val로 사용), imgsz 640, epochs 28, batch 16, seed 0. val mAP50 0.957.

### 다시 만들 때 (순서대로)

1. `data/split.csv`를 아래 형식대로 저장
2. 계획 확인 (아무것도 쓰지 않음)

   ```
   .venv/bin/python code/train_A.py --plan
   ```

   split 값, image_id 6자리·중복, 한 segment가 여러 split에 걸치는지, A·B는 LLVIP train·test는 LLVIP test에 있는지 검사합니다. 통과하면 이미지 수, val로 뗄 segment, 저장될 파일 이름을 보여 줍니다.

3. 평균 융합 → A로 학습 → B·Test 후보 저장 (한 번에)

   ```
   .venv/bin/python code/train_A.py
   ```

| 단계 | 내용 | 출력 |
| --- | --- | --- |
| 평균 융합 | A·B는 LLVIP train, test는 LLVIP test에서 RGB와 열화상을 0.5:0.5로 평균 | `data/fused_avg/{A,B,test}/` |
| 데이터셋 | XML → YOLO 라벨, person 1클래스 (이미지 밖 좌표는 잘라냄) | `data/yolo_A/` |
| val 분리 | A 안에서 segment 단위로 뗌. seed 0으로 segment를 섞어 A 이미지의 10% 이상이 될 때까지. B와 test는 학습에도 val에도 쓰지 않음 | `models/A_avg/A_val_segments.csv` |
| 학습 | yolo11n.pt(COCO), imgsz 640, epochs 28, seed 0, deterministic, batch 16 | `models/A_avg/weights/best.pt` |
| 후보 저장 | conf 0.001, iou 0.7, imgsz 640, max_det 1000 | `candidates/candidates_avg_B.csv`, `candidates/DO_NOT_OPEN_candidates_avg_test.csv` |

- 마지막에 후보 파일별로 이미지 수, 후보 수, 이미지당 최대 후보 수가 출력됩니다. ⚠ `max_det 1000에 닿은 이미지 있음`이 보이면 후보가 잘렸을 수 있습니다.
- 후보 파일이 이미 있으면 덮어쓰지 않고 멈춥니다. 덮어쓰려면 `--overwrite`
- 자주 쓰는 옵션: `--name`(저장 폴더, 기본 A_avg), `--model-tag`(후보 파일 이름의 모델 부분, 기본 avg, 예: 4ch), `--val-frac`, `--val-seed`, `--skip-train --weights <best.pt>`(학습 없이 후보만 다시 저장)

## 파일 형식

### 후보 CSV

파일 이름: `candidates/candidates_{모델}_{split}.csv` (모델: proxy / avg / 4ch, split: B / test)

```
image_id,x1,y1,x2,y2,conf
010001,1023.377258,331.265564,1142.373291,570.388062,0.835560
```

| 열 | 설명 |
| --- | --- |
| image_id | 6자리 문자열 (010001처럼 앞의 0 유지). pandas로 읽을 때 `dtype={"image_id": str}` |
| x1, y1, x2, y2 | 박스 좌상단·우하단, 원본 1280×1024 픽셀 좌표 (640으로 줄인 좌표가 아님) |
| conf | 점수. conf 0.001 이상 후보를 모두 저장 |

### split.csv

경로: `data/split.csv`

```
image_id,segment_id,split
010001,1,A
030001,3,B
190001,19,test
```

| 열 | 설명 |
| --- | --- |
| image_id | 6자리 문자열. A·B는 LLVIP train, test는 LLVIP test의 파일 이름 |
| segment_id | 장면 번호 (파일 이름 앞 두 자리). 같은 segment는 반드시 같은 split에 넣습니다. E3·E4 교차검증의 fold도 segment 단위로 나눕니다 |
| split | 값은 정확히 `A` / `B` / `test` 중 하나 (대소문자 구분, 앞뒤 공백 없음). 코드는 `split == "B"`로 B 이미지를 고릅니다 |

## 분석 실행

```
.venv/bin/python code/run_E1.py candidates/candidates_avg_B.csv
.venv/bin/python code/run_E2.py candidates/candidates_avg_B.csv
.venv/bin/python code/run_E3.py candidates/candidates_avg_B.csv   # --mode insample|cv|both (기본 both)
.venv/bin/python code/run_E4.py                                   # E4-0, E4-1 (avg_B 고정)
.venv/bin/python code/run_E4_2.py                                 # E4-2 (run_E4.py의 특징 캐시를 재사용)
.venv/bin/python code/plot_slides.py avg_B                        # 발표용 그래프 2장
```

- 결과 파일 이름에는 후보 파일의 모델·split이 들어갑니다 (예: `results/E1/E1_curve_all_avg_B.png`).
- E3·E4는 B 후보로만 실행됩니다. E4는 E3가 만든 fold 배정 파일(`results/E3/E3_cv_fold_assignment_avg_B.csv`)을 그대로 쓰므로 **E3를 먼저 실행**해야 합니다.
- E4-1은 처음 실행할 때 열화상·RGB 이미지를 읽어 특징을 뽑고 `data/features_avg_B.parquet`에 저장합니다 (이미지당 약 31ms). 이후에는 캐시를 씁니다. E4-2의 추가 특징은 `data/features_avg_B_e42.parquet`에 저장됩니다.
- 결과 폴더를 바꾸려면 환경변수 `LLVIP_RESULTS`를 지정합니다 (테스트용).
- proxy(Test) 결과는 코드 검증용입니다. csv 첫 줄에 그렇게 표시되고, 발표 수치로 쓰지 않습니다.

### E4 특징 묶음

| 묶음 | 특징 |
| --- | --- |
| ① 박스 | conf, 높이, 너비, 종횡비, y중심, 다른 후보(conf ≥ 0.1)와의 최대 IoA, 주변 후보 수, 이미지 내 conf 순위 |
| D 중복 | 나보다 conf 높은 후보와의 최대 IoU, 그 후보와의 conf 비율, 겹친 후보 묶음 안에서의 순위 |
| ② 열화상 | 박스 안 평균·최대 밝기, 박스 안 − 테두리(폭의 20%) 밝기, 세로 3등분 평균, 밝은 픽셀 비율(이미지 80분위 기준) |
| ③ RGB | 박스 안 밝기 평균·표준편차, 박스 안 − 테두리 밝기 |
| E4-2 변경 | y중심 제거, logit(conf) 추가, ②③은 상대값만 (테두리 차이, 이미지 표준편차로 나눈 값, 이미지 평균 대비 비율) |

### 테스트

```
.venv/bin/python code/test_eval.py
.venv/bin/python code/test_E3.py
.venv/bin/python code/test_E4.py
```

## 주요 결과 요약 (B, FPPI 0.5)

| 실험 | 결과 |
| --- | --- |
| E1 | Recall 전체 0.873 / Q1 0.685 / 겹침 0.751. 놓친 정답의 72%가 low_conf(후보는 있으나 점수 부족). 작은 사람은 놓친 경우의 36%가 후보 없음 |
| E2 | 겹친 후보는 conf 0.1 이상 모든 구간에서 같은 점수의 실제 사람 비율이 더 낮음 (중복 박스가 섞임). 크기 치우침은 일부 구간에만 |
| E3 | 그룹별 합격선(5-fold): Q1 +0.92 ± 0.71%p, 겹침 −1.26 ± 0.70%p, 전체 +0.16 ± 0.15%p |
| E4-0 | 후보가 있는 정답을 모두 살리면 (conf ≥ 0.01 후보 기준) Q1 +15.2%p, 겹침 +15.3%p까지 가능 |
| E4-1 | LightGBM 재점수화는 네 구성 모두 원래 점수보다 낮음. 로지스틱 회귀만 소폭 상승(표준편차 안) |
| E4-2 | y중심 하나만 빼도 LightGBM 손실이 크게 줄어(Q1 −3.18 → −1.09%p) 장면 외우기 가설 지지. 원래 점수보다 확실히 나은 모델은 없음 |

자세한 수치는 `results/` 아래 csv를 참고하세요.

## 다음 단계: ResNet 재판별기 (code/resnet_crop.py)

- `CropDataset`: 후보 박스에 여백 20%를 붙여 일반 사진·열화상에서 같은 자리를 자르고, 112×112로 줄여 R, G, B, 열화상 4채널 텐서로 반환. 라벨은 정답과 IoU ≥ 0.5면 1
- `make_resnet18_4ch()`: ImageNet 사전학습 ResNet18의 첫 층을 4채널로 바꿈 (열화상 채널은 RGB 가중치 평균으로 초기화), 출력 1개(logit)
- 계획: `weights/A_avg_best.pt`로 **A 이미지의 후보**를 뽑아 그 crop으로 학습하고, B 전체로 평가 (장면 13개에서 배우고 B는 깨끗한 평가용으로)
- 서버에서 쓸 때 주의:
  - 파일 맨 위의 `ROOT = '/content/LLVIP'`는 Colab 경로입니다. 서버에서는 `data/LLVIP` 경로로 바꿔야 합니다.
  - `CropDataset`의 라벨은 "정답과 IoU ≥ 0.5인지"라서, E2~E4가 쓰는 1:1 매칭 라벨(같은 사람의 두 번째 박스는 0)과 다릅니다. 평가와 합칠 때는 라벨 기준을 맞춰야 합니다.

## 저장소에 올리지 않는 것

- 데이터셋(`data/LLVIP`), 융합 이미지, YOLO 데이터셋, 특징 캐시, 가상환경(`.venv`), 학습 중간 산출물(`models/`)
- **Test 후보(`DO_NOT_OPEN_…`)는 열어 보지 말고 커밋하지 마세요.** 결정과 발표 수치는 B 후보로만 냅니다. Test는 최종 평가 때 한 번만 엽니다.
- `.pt` 파일은 기본적으로 막혀 있고, `weights/` 폴더만 예외로 올라갑니다.
