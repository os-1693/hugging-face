"""
🔮 簡単予測スクリプト - 学習したモデルを使ってみよう！

このスクリプトは、学習済みモデルを使って感情分析を簡単に試せます。

実行方法:
    python quickstart_predict.py

前提条件:
    - quickstart_simple.py を実行済み
    - models/my-first-model/ にモデルが保存されている
"""

import os
import sys

print("=" * 60)
print("🔮 感情分析AIで遊んでみよう！")
print("=" * 60)
print()

# =============================================================================
# モデルの確認
# =============================================================================
model_path = "./models/my-first-model"

if not os.path.exists(model_path):
    print("❌ エラー: モデルが見つかりません")
    print()
    print("まず以下のコマンドでモデルを学習してください:")
    print("  python quickstart_simple.py")
    print()
    sys.exit(1)

print(f"✓ モデルを発見: {model_path}")
print()

# =============================================================================
# ライブラリの読み込み
# =============================================================================
print("📦 AIを起動しています...")

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except ImportError:
    print("❌ エラー: 必要なライブラリがインストールされていません")
    print("以下のコマンドを実行してください:")
    print("  pip install -r requirements.txt")
    sys.exit(1)

# =============================================================================
# モデルの読み込み
# =============================================================================
print("🤖 モデルを読み込んでいます...")

try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    print("✓ モデルの読み込み完了！")
    print()
except Exception as e:
    print(f"❌ エラー: モデルの読み込みに失敗しました")
    print(f"エラー詳細: {e}")
    sys.exit(1)

# =============================================================================
# 予測関数
# =============================================================================
def predict_sentiment(text):
    """
    テキストの感情を予測する関数

    Args:
        text: 分析したいテキスト

    Returns:
        sentiment: "ポジティブ" または "ネガティブ"
        confidence: 確信度（0〜1）
        positive_prob: ポジティブの確率
        negative_prob: ネガティブの確率
    """
    # トークナイズ
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=True
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # 予測
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

    negative_prob = probabilities[0][0].item()
    positive_prob = probabilities[0][1].item()

    if positive_prob > negative_prob:
        sentiment = "ポジティブ"
        confidence = positive_prob
        emoji = "😊"
    else:
        sentiment = "ネガティブ"
        confidence = negative_prob
        emoji = "😔"

    return sentiment, confidence, positive_prob, negative_prob, emoji

# =============================================================================
# サンプル予測
# =============================================================================
print("=" * 60)
print("📝 サンプルテキストで試してみましょう")
print("=" * 60)
print()

sample_texts = [
    "This movie was absolutely fantastic! Best film I've seen this year!",
    "Terrible waste of time. I want my money back.",
    "It was okay, nothing special.",
    "Amazing performance by the actors. Highly recommended!",
    "Boring and predictable. Fell asleep halfway through.",
]

for i, text in enumerate(sample_texts, 1):
    sentiment, confidence, pos_prob, neg_prob, emoji = predict_sentiment(text)

    print(f"サンプル {i}:")
    print(f"  入力: \"{text}\"")
    print(f"  結果: {sentiment} {emoji} (確信度: {confidence:.1%})")
    print(f"  詳細: ネガティブ {neg_prob:.1%} | ポジティブ {pos_prob:.1%}")
    print()

# =============================================================================
# インタラクティブモード
# =============================================================================
print("=" * 60)
print("✨ あなたのテキストで試してみましょう！")
print("=" * 60)
print()
print("💡 ヒント:")
print("  - 英語のテキストを入力してください")
print("  - 映画レビューのような文章が最適です")
print("  - 'quit' または 'exit' で終了")
print()

while True:
    try:
        # ユーザー入力
        user_input = input("📝 あなたのテキスト: ").strip()

        # 終了コマンドのチェック
        if user_input.lower() in ['quit', 'exit', 'q', '終了']:
            print()
            print("👋 ありがとうございました！またね！")
            break

        # 空入力のチェック
        if not user_input:
            print("⚠️  テキストを入力してください")
            print()
            continue

        # 予測実行
        sentiment, confidence, pos_prob, neg_prob, emoji = predict_sentiment(user_input)

        # 結果表示
        print()
        print(f"🔮 予測結果:")
        print(f"  感情: {sentiment} {emoji}")
        print(f"  確信度: {confidence:.1%}")
        print()

        # 詳細な分析
        print(f"📊 詳細:")
        print(f"  ネガティブ: {'█' * int(neg_prob * 20)} {neg_prob:.1%}")
        print(f"  ポジティブ: {'█' * int(pos_prob * 20)} {pos_prob:.1%}")
        print()

        # 解釈のヘルプ
        if confidence >= 0.9:
            print("💡 解釈: AIはかなり確信を持っています")
        elif confidence >= 0.7:
            print("💡 解釈: AIはある程度確信しています")
        else:
            print("💡 解釈: AIは少し迷っているようです（中立的な文章かも）")

        print()
        print("-" * 60)
        print()

    except KeyboardInterrupt:
        print()
        print()
        print("👋 中断されました。またね！")
        break
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        print()

# =============================================================================
# 終了メッセージ
# =============================================================================
print()
print("=" * 60)
print("✅ 予測スクリプト終了")
print("=" * 60)
print()
print("🎯 次にやってみること:")
print("  1. より長く学習してみる:")
print("     - config/beginner_config.yaml で num_epochs を増やす")
print()
print("  2. 別のデータセットを試す:")
print("     - 設定ファイルの dataset.name を変更")
print()
print("  3. 日本語モデルで試す:")
print("     - model.name を 'cl-tohoku/bert-base-japanese' に変更")
print()
print("📚 詳しい情報:")
print("  - QUICKSTART.md")
print("  - docs/CONCEPTS.md")
print("=" * 60)
