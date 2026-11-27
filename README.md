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
| `code/train_A_model.py` | A로 탐지기 학습, 베이스 가중치를 인자로 받음 (`--base yolo11s.pt`). 학습 데이터·val 분리·설정은 `train_A.py`와 같고 Test는 다루지 않음. 학습 시간을 `models/<name>/train_time.json`에 기록 |
| `code/run_yolo_s.sh` | YOLO11s 학습 → B 후보 생성 → (학습 시간 비교용) YOLO11n 재학습을 순서대로 실행 (nohup으로 돌리는 용도) |
| `code/run_yolo_s.py` | YOLO11n vs YOLO11s 평가 (B, E1과 같은 방식). n 수치가 기존 E1 결과와 같은지 확인하고 `results/yolo_s/`에 저장 |
| `code/speed_yolo.py` | n·s 추론 속도 비교 (같은 GPU, imgsz 640, batch 1, warmup 후 B 500장), 파라미터 수·GFLOPs |

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

### YOLO11s 비교 (YOLO11n과 같은 설정)

```
nohup code/run_yolo_s.sh > models/yolo_s_pipeline.log 2>&1 &   # s 학습 → s B 후보 → n 재학습(시간 측정용)
.venv/bin/python code/run_yolo_s.py                              # n vs s 평가 → results/yolo_s/
.venv/bin/python code/speed_yolo.py                              # 속도 → results/yolo_s/speed.csv (GPU가 비어 있을 때)
```

- s는 `train_A.py`와 같은 데이터(A, 평균 융합)·val 분리·설정(imgsz 640, epochs 28, batch 16, seed 0)에 베이스만 `yolo11s.pt`로 바꿔 학습합니다. 가중치는 `weights/A_avg_s_best.pt`, B 후보는 `candidates/candidates_avg_s_B.csv`
- n 지표·속도는 기존 `weights/A_avg_best.pt`와 E1 후보(`candidates_avg_B.csv`)를 씁니다. n 재학습(`models/A_avg_n_retime/`)은 같은 GPU에서 학습 시간을 재기 위한 것이라 저장소에 올리지 않습니다

## 주요 결과 요약 (B, FPPI 0.5)

| 실험 | 결과 |
| --- | --- |
| E1 | Recall 전체 0.873 / Q1 0.685 / 겹침 0.751. 놓친 정답의 72%가 low_conf(후보는 있으나 점수 부족). 작은 사람은 놓친 경우의 36%가 후보 없음 |
| E2 | 겹친 후보는 conf 0.1 이상 모든 구간에서 같은 점수의 실제 사람 비율이 더 낮음 (중복 박스가 섞임). 크기 치우침은 일부 구간에만 |
| E3 | 그룹별 합격선(5-fold): Q1 +0.92 ± 0.71%p, 겹침 −1.26 ± 0.70%p, 전체 +0.16 ± 0.15%p |
| E4-0 | 후보가 있는 정답을 모두 살리면 (conf ≥ 0.01 후보 기준) Q1 +15.2%p, 겹침 +15.3%p까지 가능 |
| E4-1 | LightGBM 재점수화는 네 구성 모두 원래 점수보다 낮음. 로지스틱 회귀만 소폭 상승(표준편차 안) |
| E4-2 | y중심 하나만 빼도 LightGBM 손실이 크게 줄어(Q1 −3.18 → −1.09%p) 장면 외우기 가설 지지. 원래 점수보다 확실히 나은 모델은 없음 |
| YOLO11s | 같은 설정에서 탐지기만 YOLO11s로: Recall 전체 0.912 / Q1 0.753 / 겹침 0.798 (n 대비 +3.9 / +6.8 / +4.7%p), conf 0.25 FPPI 0.616 → 0.338. 개선은 low_conf 감소에서 나오고 후보 없음(no_candidate)은 줄지 않음(겹침 5.8 → 7.3%). batch 1 추론은 n과 같음(A5000 약 5.8ms), 학습 15.9 → 18.4분 |

자세한 수치는 `results/` 아래 csv를 참고하세요.

## 결과 파일 설명

모든 결과는 B(avg 후보) 기준입니다. `*_proxy_test.*`는 코드 검증용(proxy 재학습, Test)이라 발표 수치로 쓰지 않습니다.
일부 csv는 첫 줄이 설명 문구이므로 `pd.read_csv(path, skiprows=1)`로 읽습니다.

### results/E1/ — FPPI-Recall, 미탐 원인

| 파일 | 내용 | 주요 열 |
| --- | --- | --- |
| `E1_curve_avg_B.csv` | 합격선 0.001~0.95 (0.001 간격) 스윕 | `thr, tp, fp, fppi, recall, recall_q1, recall_overlap` |
| `E1_recall_at_fppi_avg_B.csv` | FPPI 0.25/0.5/1.0에서 Recall (선형 보간) | `fppi, all, q1, overlap` |
| `E1_miss_causes_avg_B.csv` | conf 0.25 기준 미탐 원인 비율 | `group, n_gt, hit, low_conf, no_candidate` |
| `E1_curve_{all,q1,overlap}_avg_B.png` | FPPI-Recall 곡선 (FPPI 0~2) | |
| `E1_miss_causes_avg_B.png` | 미탐 원인 가로 누적 막대 | |

### results/E2/ — 점수 치우침

| 파일 | 내용 | 주요 열 |
| --- | --- | --- |
| `E2_reliability_avg_B.csv` | conf 구간·그룹별 사람 비율 | `group_type, group, bin, n, person_ratio, mean_conf, low_n` |
| `E2_reliability_{size,overlap}_avg_B.png` | 크기별·겹침별 reliability diagram (후보 50개 미만 구간 표시) | |

### results/E3/ — 그룹별 합격선

| 파일 | 내용 | 주요 열 |
| --- | --- | --- |
| `E3_cv_summary_avg_B.csv` | 5-fold 교차검증 평균 ± 표준편차 | `method, fppi, recall, recall_q1, recall_overlap` (+ `_mean`, `_std`) |
| `E3_cv_folds_avg_B.csv` | fold별 고른 합격선과 빠진 fold 평가 | `fold, t_small, t_normal_ov, t_normal_noov, group_*, single_*, diff_*` |
| `E3_cv_fold_assignment_avg_B.csv` | segment → fold 배정 | `segment_id, n_images, fold` |
| `E3_insample_{result,top20_fast,verify}_avg_B.csv` | 참고용(in-sample): B 전체로 고르고 B 전체에서 평가 | `method, t_*, fppi, recall*, objective` |

### results/E4/ — 개선 여지 상한, 재점수화

| 파일 | 내용 | 주요 열 |
| --- | --- | --- |
| `E4_0_upper_bound_avg_B.csv` | FPPI 0.5 지점 Recall 상한 | `group, recall, upper_all_cands, gain_all_%p, upper_conf>=0.01, gain_conf>=0.01_%p` |
| `E4_0_headroom_avg_B.png` | 실제 Recall vs 상한 묶음 막대 | |
| `E4_1_{folds,summary}_avg_B.csv` | E4-1 LightGBM·LR 재점수화, fold별 / 평균 ± 표준편차 | `config, fppi, new_*, base_*, diff_*` |
| `E4_1_importance_avg_B.csv` | LightGBM 특징 중요도(gain) 상위 15 | `config, feature, gain, rank` |
| `E4_E3_folds_avg_B.csv` | 재점수 + 그룹별 합격선 | `fold, t_*, fppi, recall*, diff_*` |
| `E4_compare_avg_B.csv` | E4-1 비교 요약 (FPPI 0.5) | `method, fppi, recall, recall_diff_%p, …` |
| `E4_feature_time_avg_B.csv` | 이미지당 특징 추출 시간 | `part, ms_per_image_mean, ms_per_image_median` |
| `E4_2_{folds,summary,scenes}_avg_B.csv` | E4-2 장면 의존 특징 제거, fold별 / 평균 / segment별 | `model, fppi, new_*, base_*, diff_*` (scenes는 `segment_id` 추가) |
| `E4_2_importance_avg_B.csv`, `E4_2_lr_coef_avg_B.csv` | LightGBM 중요도, LR 계수(표준화 기준) | `model, feature, gain` / `model, feature, mean, std` |
| `E4_2_compare_avg_B.csv` | E4-1 vs E4-2 (FPPI 0.5) | `실험, 모델, recall, recall_diff_%p, …` |
| `E4_meta_avg_B.json` | E4-1 설정·가장 좋은 구성 | |

### results/yolo_s/ — YOLO11n vs YOLO11s

n = `candidates/candidates_avg_B.csv` (E1과 같은 후보, `weights/A_avg_best.pt`),
s = `candidates/candidates_avg_s_B.csv` (`weights/A_avg_s_best.pt`, n과 같은 학습 설정에 베이스만 yolo11s.pt).

| 파일 | 내용 | 주요 열 |
| --- | --- | --- |
| `compare_n_vs_s_B.csv` | n·s 지표 나란히 + 마지막 행 s − n | `model, n_cands, recall@fppi{0.25,0.5,1.0}, recall_q1@…, recall_overlap@…, conf0.25_fppi, conf0.25_recall*` |
| `recall_at_fppi_avg_B.csv`, `recall_at_fppi_avg_s_B.csv` | FPPI 0.25/0.5/1.0에서 Recall (E1과 같은 형식) | `fppi, all, q1, overlap` |
| `curve_avg_B.csv`, `curve_avg_s_B.csv` | 합격선 스윕 (E1과 같은 형식) | `thr, tp, fp, fppi, recall, recall_q1, recall_overlap` |
| `miss_causes_avg_B.csv`, `miss_causes_avg_s_B.csv` | conf 0.25 미탐 원인 (E1과 같은 형식) | `group, n_gt, hit, low_conf, no_candidate` |
| `miss_causes_n_vs_s_B.csv` | 미탐 원인 n·s 한 표 | `model, group, n_gt, hit, low_conf, no_candidate` |
| `speed.csv` | 추론 속도 (A5000, imgsz 640, batch 1, B 500장, warmup 30장, n→s 2회 반복) | `repeat, model, params_M, GFLOPs, inference_ms_mean, inference_ms_median, pre+inf+post_ms_mean, wall_ms_mean(읽기 포함)` |
| `fppi_recall_n_vs_s_B.png` | 전체·Q1·겹침 FPPI-Recall 곡선 n vs s | |
| `miss_causes_n_vs_s_B.png` | 미탐 원인 가로 누적 막대 n vs s | |

## PPT ↔ repo 실험 대응표

| PPT 슬라이드 | 실험 | 코드 | 결과 폴더 |
| --- | --- | --- | --- |
|  | 공통 평가 함수·proxy 검증 | `code/eval.py`, `code/test_eval.py` | `results/E1/*_proxy_test.*` |
|  | E1 FPPI-Recall·미탐 원인 | `code/run_E1.py`, `code/plot_slides.py` | `results/E1/` |
|  | E2 점수 치우침 | `code/run_E2.py` | `results/E2/` |
|  | E3 그룹별 합격선 (5-fold) | `code/run_E3.py` | `results/E3/` |
|  | E4-0 개선 여지 상한 | `code/run_E4.py`, `code/plot_slides.py` | `results/E4/E4_0_*` |
|  | E4-1 LightGBM 재점수화 | `code/run_E4.py` | `results/E4/E4_1_*`, `E4_E3_*`, `E4_compare_*` |
|  | E4-2 장면 의존 특징 제거 | `code/run_E4_2.py` | `results/E4/E4_2_*` |
|  | YOLO11n vs YOLO11s | `code/train_A_model.py`, `code/run_yolo_s.sh`, `code/run_yolo_s.py`, `code/speed_yolo.py` | `results/yolo_s/` |

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
