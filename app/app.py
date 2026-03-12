import json

with open("metrics/metrics.json") as f:
    metrics = json.load(f)

print("Model Metrics")
print(metrics)