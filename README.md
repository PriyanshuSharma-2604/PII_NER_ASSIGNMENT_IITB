# PII NER Assignment Skeleton

This repo is a skeleton for a token-level NER model that tags PII in STT-style transcripts.

## Setup

```bash
pip install -r requirements.txt
```

## Train

```bash
python src/train.py \
  --model_name distilbert-base-uncased \
  --train data/train.jsonl \
  --dev data/dev.jsonl \
  --out_dir out
```

## Predict

```bash
python src/predict.py \
  --model_dir out \
  --input data/dev.jsonl \
  --output out/dev_pred.json
```

## Evaluate

```bash
# Dev set
python src/eval_span_f1.py \
  --gold data/dev.jsonl \
  --pred out/dev_pred.json

# (Optional) stress test set
python src/predict.py \
  --model_dir out \
  --input data/stress.jsonl \
  --output out/stress_pred.json

python src/eval_span_f1.py \
  --gold data/stress.jsonl \
  --pred out/stress_pred.json
```

## Measure latency

```bash
python src/measure_latency.py \
  --model_dir out \
  --input data/dev.jsonl \
  --runs 50
```

Your task in the assignment is to modify the model and training code to improve entity and PII detection quality while keeping **p95 latency below ~20 ms** per utterance (batch size 1, on a reasonably modern CPU).

## ONNX export & low-latency inference (recommended for CPU deployments)

After training the model, export it to ONNX and run the ONNX-based pipeline for low latency:

PowerShell quick-flow (from project root):

```powershell
# export ONNX
python src/export_onnx.py --model_dir out --input data/dev.jsonl --max_length 128

# run ONNX predictions (predict.py will automatically use model.onnx if present)
python src/predict.py --model_dir out --input data/dev.jsonl --output out/dev_pred_onnx.json

# evaluate
python src/eval_span_f1.py --gold data/dev.jsonl --pred out/dev_pred_onnx.json

# measure ONNX latency
python src/measure_latency_onnx.py --model_dir out --input data/dev.jsonl --runs 50 --max_length 128
```

The ONNX Runtime path typically produces much lower p95 latency on CPU (in my runs p95 dropped from ~28ms to ~14ms).
