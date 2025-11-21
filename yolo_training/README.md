# YOLO Training

## フォルダ構造

```
yolo_training/
├── datasets/
│   ├── images/
│   │   ├── train/    # 訓練用画像
│   │   └── val/      # 検証用画像
│   └── labels/
│       ├── train/    # 訓練用ラベル (YOLO形式 .txt)
│       └── val/      # 検証用ラベル
├── weights/          # 学習済みモデル
├── runs/             # 学習結果の出力
└── data.yaml         # データセット設定
```

## 使用方法

### 学習の実行

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # or yolov8s, yolov8m, etc.
model.train(data='yolo_training/data.yaml', epochs=100)
```

### Hugging Face Hubとの連携

```python
# モデルのアップロード
model.export(format='onnx')
# huggingface_hub を使ってHubにプッシュ可能
```

## ラベル形式

YOLO形式: `class_id x_center y_center width height` (正規化された値 0-1)
