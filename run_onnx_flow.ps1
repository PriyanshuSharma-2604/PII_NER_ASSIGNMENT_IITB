# Run the full ONNX flow: export -> predict -> eval -> latency
param(
    [string]$ModelDir = "out",
    [string]$DevJson = "data/dev.jsonl",
    [string]$StressJson = "data/stress.jsonl"
)

Write-Host "Exporting ONNX model..."
python src/export_onnx.py --model_dir $ModelDir --input $DevJson --max_length 128

Write-Host "Running ONNX predictions on dev..."
python src/predict.py --model_dir $ModelDir --input $DevJson --output $ModelDir/dev_pred_onnx.json

Write-Host "Evaluating dev predictions..."
python src/eval_span_f1.py --gold $DevJson --pred $ModelDir/dev_pred_onnx.json

Write-Host "Running ONNX predictions on stress..."
python src/predict.py --model_dir $ModelDir --input $StressJson --output $ModelDir/stress_pred_onnx.json

Write-Host "Evaluating stress predictions..."
python src/eval_span_f1.py --gold $StressJson --pred $ModelDir/stress_pred_onnx.json

Write-Host "Measuring ONNX latency (dev)..."
python src/measure_latency_onnx.py --model_dir $ModelDir --input $DevJson --runs 50 --max_length 128

Write-Host "Done."
