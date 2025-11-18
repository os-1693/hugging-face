# Hugging Face モデル学習プロジェクト

Hugging Face Transformersを使用したモデルの学習・評価を行うためのPythonプロジェクトです。

## プロジェクト構成

```
hugging-face/
├── config/              # 設定ファイル
│   └── train_config.yaml
├── data/                # データセット格納用
├── models/              # 学習済みモデル保存用
├── notebooks/           # Jupyter Notebook用
├── src/                 # ソースコード
│   ├── __init__.py
│   ├── dataset.py       # データセット処理
│   ├── model.py         # モデル定義
│   └── train.py         # 学習スクリプト
├── requirements.txt     # 依存パッケージ
└── README.md
```

## セットアップ

### 1. 仮想環境の作成（推奨）

```bash
python -m venv venv
source venv/bin/activate  # Windowsの場合: venv\Scripts\activate
```

### 2. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

## 使い方

### 1. 設定ファイルの編集

`config/train_config.yaml`を編集して、モデル、データセット、学習パラメータを設定します。

```yaml
model:
  name: "bert-base-uncased"  # 使用するモデル
  task_type: "sequence_classification"  # タスクタイプ
  num_labels: 2  # ラベル数

dataset:
  name: "imdb"  # データセット名

training:
  num_epochs: 3
  batch_size: 16
  learning_rate: 2.0e-5
```

### 2. モデルの学習

```bash
python src/train.py --config config/train_config.yaml
```

### 3. TensorBoardでの進捗確認

```bash
tensorboard --logdir ./logs
```

ブラウザで http://localhost:6006 にアクセスして学習の進捗を確認できます。

## サポートされているタスク

### 1. テキスト分類（Sequence Classification）

感情分析、トピック分類など

```python
from src.model import ModelBuilder

model = ModelBuilder.build_sequence_classification_model(
    model_name="bert-base-uncased",
    num_labels=2
)
```

### 2. トークン分類（Token Classification）

固有表現認識（NER）、品詞タグ付けなど

```python
model = ModelBuilder.build_token_classification_model(
    model_name="bert-base-uncased",
    num_labels=9
)
```

### 3. 質問応答（Question Answering）

```python
model = ModelBuilder.build_qa_model(
    model_name="bert-base-uncased"
)
```

### 4. 因果言語モデル（Causal LM）

テキスト生成など

```python
model = ModelBuilder.build_causal_lm_model(
    model_name="gpt2"
)
```

## カスタムデータセットの使用

### CSVファイルから

```python
from src.dataset import CustomDatasetBuilder

dataset = CustomDatasetBuilder.from_csv(
    train_path="data/train.csv",
    validation_path="data/validation.csv",
    test_path="data/test.csv"
)
```

### JSONファイルから

```python
dataset = CustomDatasetBuilder.from_json(
    train_path="data/train.json",
    validation_path="data/validation.json",
    test_path="data/test.json"
)
```

## Weights & Biases (wandb) の使用

学習の追跡にwandbを使用する場合：

1. wandbにログイン

```bash
wandb login
```

2. 設定ファイルを編集

```yaml
training:
  report_to: ["wandb"]
```

## 推奨モデル

### 日本語タスク

- `bert-base-japanese`
- `cl-tohoku/bert-base-japanese-v2`
- `rinna/japanese-gpt-1b`

### 英語タスク

- `bert-base-uncased` (テキスト分類、NER)
- `roberta-base` (より高性能なBERT変種)
- `gpt2` (テキスト生成)
- `distilbert-base-uncased` (軽量版)

## トラブルシューティング

### CUDAエラー

GPUが利用できない場合、設定ファイルで`fp16: false`に設定してください。

### メモリ不足

バッチサイズを小さくするか、勾配累積を使用してください：

```yaml
training:
  batch_size: 8
  gradient_accumulation_steps: 2
```

### データセットが見つからない

Hugging Face Hubでデータセット名を確認してください：
https://huggingface.co/datasets

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 参考資料

- [Hugging Face Transformers ドキュメント](https://huggingface.co/docs/transformers)
- [Hugging Face Datasets](https://huggingface.co/docs/datasets)
- [Trainer API](https://huggingface.co/docs/transformers/main_classes/trainer)
