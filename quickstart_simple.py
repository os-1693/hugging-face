"""
超簡単！初めてのAIモデル学習

このスクリプトは、AI学習が初めての方でも簡単に実行できるように作られています。
実行するだけで、感情分析モデル（ポジティブ/ネガティブ判定）の学習ができます。

実行方法:
    python quickstart_simple.py

所要時間: 約3〜5分（CPUの場合は10〜15分）
"""

import os
import sys

# =============================================================================
# ステップ1: 必要なライブラリのインポート
# =============================================================================
print("=" * 60)
print("AI学習を始めます")
print("=" * 60)
print("\nステップ1: ライブラリを読み込んでいます...")

try:
    import numpy as np
    import torch
    from datasets import load_dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    print("ライブラリの読み込みが完了しました")
except ImportError as e:
    print(f"エラー: 必要なライブラリがインストールされていません")
    print(f"   以下のコマンドを実行してください:")
    print(f"   pip install -r requirements.txt")
    sys.exit(1)

# 再現性のための乱数シード設定
set_seed(42)

# =============================================================================
# ステップ2: 使用するデバイス（GPU or CPU）の確認
# =============================================================================
print("\nステップ2: 使用するデバイスを確認しています...")

# GPUが利用可能でも古い場合、CPUを使用
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPUが検出されました: {gpu_name}")
    # CUDA capabilityを確認（簡易チェック）
    try:
        # PyTorchがサポートする最小capabilityは7.0
        # 古いGPUの場合、CPUを使用
        device = torch.device("cpu")
        print(
            "GPUが古いため、CPUを使用します（学習時間が長くなりますが問題ありません）"
        )
    except:
        device = torch.device("cpu")
        print("GPUの互換性に問題があるため、CPUを使用します")
else:
    device = torch.device("cpu")
    print("GPUが利用できないため、CPUを使用します")

print(f"最終デバイス: {device}")

# =============================================================================
# ステップ3: データセットの準備
# =============================================================================
print("\nステップ3: データセットをダウンロードしています...")
print("   （初回のみ時間がかかります）")

try:
    # IMDbの映画レビューデータセット（感情分析用）
    # ポジティブなレビューとネガティブなレビューが含まれています
    # 初心者向けに少量のデータのみ使用（学習時間を短縮）

    # データセット全体をロードしてシャッフル
    full_dataset = load_dataset("imdb")
    shuffled_train = full_dataset["train"].shuffle(seed=42)
    shuffled_test = full_dataset["test"].shuffle(seed=42)

    # バランスの取れたデータセットを作成
    dataset = {
        "train": shuffled_train.select(range(5000)),
        "test": shuffled_test.select(range(1000)),
    }

    print(f"データセットのダウンロード完了")
    print(f"  - 訓練データ: {len(dataset['train'])}件")
    print(f"  - テストデータ: {len(dataset['test'])}件")

    # データの例を表示
    print(f"\nデータの例:")
    example = dataset["train"][0]
    print(f"  レビュー: {example['text'][:100]}...")
    print(f"  感情: {'ポジティブ' if example['label'] == 1 else 'ネガティブ'}")

except Exception as e:
    print(f"エラー: データセットのダウンロードに失敗しました")
    print(f"   インターネット接続を確認してください")
    print(f"   エラー詳細: {e}")
    sys.exit(1)

# =============================================================================
# ステップ4: AIモデルの準備
# =============================================================================
print("\nステップ4: AIモデルを準備しています...")

# 使用するモデル: DistilBERT（BERTの軽量版で初心者におすすめ）
model_name = "distilbert-base-uncased"
print(f"   使用モデル: {model_name}")

try:
    # トークナイザー: テキストをAIが理解できる数値に変換するツール
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"トークナイザーの読み込み完了")

    # モデル: 実際に感情分析を行うAI
    # num_labels=2 は「ポジティブ」と「ネガティブ」の2クラス
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)
    print(f"モデルの読み込み完了")

except Exception as e:
    print(f"エラー: モデルの読み込みに失敗しました")
    print(f"   エラー詳細: {e}")
    sys.exit(1)

# =============================================================================
# ステップ5: データの前処理
# =============================================================================
print("\nステップ5: データを前処理しています...")


def preprocess_function(examples):
    """
    テキストをAIが理解できる形式に変換する関数

    例: "This movie is great!" → [101, 2023, 3185, 2003, 2307, 999, 102]
    """
    return tokenizer(
        examples["text"],
        padding="max_length",  # 全て同じ長さに揃える
        truncation=True,  # 長すぎる文章は切り詰める
        max_length=256,  # 最大256トークン（初心者向けに短めに設定）
    )


try:
    # データセット全体に前処理を適用
    tokenized_train = dataset["train"].map(
        preprocess_function,
        batched=True,
        remove_columns=["text"],  # 元のテキストは不要なので削除
    )
    tokenized_test = dataset["test"].map(
        preprocess_function,
        batched=True,
        remove_columns=["text"],  # 元のテキストは不要なので削除
    )
    tokenized_dataset = {"train": tokenized_train, "test": tokenized_test}
    print(f"データの前処理が完了しました")

except Exception as e:
    print(f"エラー: データの前処理に失敗しました")
    print(f"   エラー詳細: {e}")
    sys.exit(1)

# =============================================================================
# ステップ6: 評価指標の設定
# =============================================================================
print("\nステップ6: 評価指標を設定しています...")


def compute_metrics(eval_pred):
    """
    モデルの性能を評価する関数
    正解率（Accuracy）を計算します
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}


print(f"評価指標の設定完了")

# =============================================================================
# ステップ7: 学習の設定
# =============================================================================
print("\nステップ7: 学習の設定をしています...")

# 学習結果の保存先
output_dir = "./models/my-first-model"
os.makedirs(output_dir, exist_ok=True)

# 学習パラメータの設定
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=1,  # 学習の繰り返し回数（1回で高速化）
    per_device_train_batch_size=8,  # 一度に処理するデータ数（小さめで安全）
    per_device_eval_batch_size=16,  # 評価時の処理数
    learning_rate=2e-5,  # 学習速度（この値が一般的）
    weight_decay=0.01,  # 過学習を防ぐパラメータ
    eval_strategy="epoch",  # 各エポック後に評価
    save_strategy="epoch",  # 各エポック後に保存
    logging_dir="./logs",  # ログの保存先
    logging_steps=50,  # 50ステップごとにログ出力
    load_best_model_at_end=True,  # 最も良いモデルを保存
    report_to=["tensorboard"],  # TensorBoardで可視化
    use_cpu=True,  # CPUを使用（最新の使い方）
)

print(f"学習設定完了")
print(f"  - エポック数: 2回")
print(f"  - バッチサイズ: 8")
print(f"  - 保存先: {output_dir}")

# =============================================================================
# ステップ8: 学習の開始
# =============================================================================
print("\n" + "=" * 60)
print("ステップ8: 学習を開始します")
print("=" * 60)
print("予想時間: 3〜5分（CPUの場合は10〜15分）")
print("学習中は休憩してお待ちください")
print()

try:
    # Trainer: 学習を自動で行ってくれる便利なツール
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        compute_metrics=compute_metrics,
    )

    # 学習開始！
    train_result = trainer.train()

    print("\n" + "=" * 60)
    print("学習が完了しました")
    print("=" * 60)

except Exception as e:
    print(f"\nエラー: 学習中にエラーが発生しました")
    print(f"   エラー詳細: {e}")
    sys.exit(1)

# =============================================================================
# ステップ9: モデルの保存
# =============================================================================
print("\nステップ9: モデルを保存しています...")

try:
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    print(f"モデルの保存完了: {output_dir}")

except Exception as e:
    print(f"エラー: モデルの保存に失敗しました")
    print(f"   エラー詳細: {e}")

# =============================================================================
# ステップ10: 結果の確認
# =============================================================================
print("\nステップ10: 結果を確認しています...")

try:
    # テストデータで評価
    eval_result = trainer.evaluate()

    print("\n" + "=" * 60)
    print("最終結果")
    print("=" * 60)
    print(f"正解率（Accuracy）: {eval_result['eval_accuracy']:.2%}")
    print(f"損失（Loss）: {eval_result['eval_loss']:.4f}")
    print()

    # 結果の解釈
    accuracy = eval_result["eval_accuracy"]
    if accuracy >= 0.85:
        print("素晴らしい！非常に高い精度です")
    elif accuracy >= 0.75:
        print("良い結果です")
    elif accuracy >= 0.65:
        print("まずまずの結果です。エポック数を増やすと改善するかもしれません。")
    else:
        print("もう少し改善の余地がありそうです。")

except Exception as e:
    print(f"エラー: 評価に失敗しました")
    print(f"   エラー詳細: {e}")

# =============================================================================
# ステップ11: 実際に使ってみよう
# =============================================================================
print("\n" + "=" * 60)
print("ステップ11: モデルを実際に使ってみましょう")
print("=" * 60)

# テスト用のサンプル文章
test_texts = [
    "This movie was absolutely fantastic! I loved every minute of it.",
    "Terrible movie. Complete waste of time and money.",
    "It was okay, nothing special but not bad either.",
]

print("\n予測を実行中...\n")

for i, text in enumerate(test_texts, 1):
    # テキストをトークナイズ
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # 予測
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

    # 結果を表示
    negative_prob = predictions[0][0].item()
    positive_prob = predictions[0][1].item()
    sentiment = "ポジティブ" if positive_prob > negative_prob else "ネガティブ"
    confidence = max(negative_prob, positive_prob)

    print(f"例 {i}:")
    print(f"  入力: {text}")
    print(f"  予測: {sentiment}")
    print(f"  確信度: {confidence:.1%}")
    print(f"  詳細: ネガティブ {negative_prob:.1%} | ポジティブ {positive_prob:.1%}")
    print()

# =============================================================================
# 完了
# =============================================================================
print("=" * 60)
print("すべて完了しました。おめでとうございます")
print("=" * 60)
print()
print("学習したモデルの場所:")
print(f"   {output_dir}")
print()
print("次にやってみること:")
print("   1. TensorBoardで学習の様子を確認:")
print("      tensorboard --logdir ./logs")
print()
print("   2. 自分のテキストで予測してみる:")
print(f"      python src/inference.py --model_path {output_dir} \\")
print('        --text "Your text here"')
print()
print("   3. より長く学習してみる:")
print("      config/beginner_config.yaml の num_epochs を 5 に変更")
print()
print("詳しい情報:")
print("   - クイックスタートガイド: QUICKSTART.md")
print("   - 基本概念の説明: docs/CONCEPTS.md")
print("=" * 60)
