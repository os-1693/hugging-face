"""
サンプル学習ノートブック（Pythonスクリプト版）

このスクリプトは、Hugging Faceでのモデル学習の基本的な流れを示しています。
Jupyter Notebookで実行することを想定していますが、Pythonスクリプトとしても実行可能です。
"""

# %% ライブラリのインポート
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# %% デバイスの確認
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# %% データセットの読み込み
print("Loading dataset...")
dataset = load_dataset("imdb", split={"train": "train[:1000]", "test": "test[:200]"})
print(f"Train dataset size: {len(dataset['train'])}")
print(f"Test dataset size: {len(dataset['test'])}")

# データの確認
print("\nExample data:")
print(dataset['train'][0])

# %% トークナイザーの読み込み
model_name = "distilbert-base-uncased"
print(f"\nLoading tokenizer: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# %% データの前処理
def preprocess_function(examples):
    """データの前処理関数"""
    return tokenizer(
        examples['text'],
        padding="max_length",
        truncation=True,
        max_length=256  # 高速化のため短めに設定
    )

print("Preprocessing dataset...")
tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=['text']
)

# %% モデルの読み込み
print(f"\nLoading model: {model_name}")
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)
model.to(device)

# %% 評価メトリクスの定義
def compute_metrics(eval_pred):
    """評価メトリクスを計算"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary'
    )

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# %% 学習設定
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=2,  # デモ用に少なめ
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_dir='./logs',
    logging_steps=50,
    report_to=["tensorboard"],
)

# %% Trainerの初期化
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test'],
    compute_metrics=compute_metrics,
)

# %% 学習の実行
print("\nStarting training...")
train_result = trainer.train()

# %% 学習結果の表示
print("\nTraining completed!")
print(f"Training metrics: {train_result.metrics}")

# %% 評価の実行
print("\nEvaluating on test set...")
eval_result = trainer.evaluate()
print(f"Evaluation metrics: {eval_result}")

# %% モデルの保存
output_dir = "./models/example-model"
print(f"\nSaving model to {output_dir}")
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print("Model saved successfully!")

# %% 推論のテスト
print("\nTesting inference...")
test_texts = [
    "This movie was absolutely fantastic! I loved every minute of it.",
    "Terrible movie. Complete waste of time."
]

# トークナイズ
inputs = tokenizer(test_texts, padding=True, truncation=True, return_tensors="pt")
inputs = {key: value.to(device) for key, value in inputs.items()}

# 推論
model.eval()
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

# 結果の表示
for text, pred in zip(test_texts, predictions):
    print(f"\nText: {text}")
    print(f"Negative: {pred[0]:.4f}, Positive: {pred[1]:.4f}")
    print(f"Predicted: {'Positive' if pred[1] > pred[0] else 'Negative'}")

print("\n✓ Example training completed successfully!")
