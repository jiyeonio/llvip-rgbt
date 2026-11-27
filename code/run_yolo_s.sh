#!/bin/bash
# YOLO11s 학습 -> B 후보 -> (학습 시간 비교용) n 재학습. 순차 실행.
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
echo "== start $(date -Is)"; nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
$PY code/train_A_model.py --base yolo11s.pt --name A_avg_s --weights-out weights/A_avg_s_best.pt
echo "== s train done $(date -Is)"
$PY code/predict.py weights/A_avg_s_best.pt data/fused_avg/B candidates/candidates_avg_s_B.csv
echo "== s predict B done $(date -Is)"
$PY code/train_A_model.py --base yolo11n.pt --name A_avg_n_retime
echo "== n retrain (time only) done $(date -Is)"
