# 🤖 Hugging Face モデル学習プロジェクト - 初心者向け

**AI学習が初めての方でも大丈夫！** このプロジェクトは、Hugging Face Transformersを使って簡単にAIモデルを学習できるように設計されています。

## 🎯 このプロジェクトでできること

- ✅ 感情分析（映画レビューがポジティブかネガティブか判定）
- ✅ テキスト分類（ニュース記事のカテゴリ分け）
- ✅ 固有表現認識（人名や地名の抽出）
- ✅ 質問応答（文章から質問に答える）
- ✅ 自分のデータでカスタムモデルを作成

## 🚀 5分でスタート！（超簡単モード）

AI学習が初めての方は、まずこれを試してください：

```bash
# 1. 必要なライブラリをインストール
pip install -r requirements.txt

# 2. たった1つのコマンドで学習開始！
python quickstart_simple.py
```

これだけで、感情分析AIモデルが完成します！（所要時間: 3〜5分）

詳しい手順は **[クイックスタートガイド](QUICKSTART.md)** をご覧ください。

## 📚 初心者向けガイド

| ドキュメント | 内容 | おすすめ度 |
|------------|------|-----------|
| [QUICKSTART.md](QUICKSTART.md) | 5分で始める超簡単ガイド | ⭐⭐⭐ 必読 |
| [docs/CONCEPTS.md](docs/CONCEPTS.md) | AI学習の基本概念を理解 | ⭐⭐⭐ 必読 |
| [docs/FAQ.md](docs/FAQ.md) | よくある質問と答え | ⭐⭐ 困った時に |

## 📁 プロジェクト構成

```
hugging-face/
├── 🚀 quickstart_simple.py      # 超簡単！これだけで学習できる
├── 🔮 quickstart_predict.py     # 学習したモデルで予測
├── 📖 QUICKSTART.md            # 5分で始めるガイド
├── config/                     # 設定ファイル
│   ├── train_config.yaml       # 上級者向け設定
│   └── beginner_config.yaml    # 初心者向け設定（詳しい説明付き）
├── docs/                       # ドキュメント
│   ├── CONCEPTS.md             # AI学習の基本概念
│   └── FAQ.md                  # よくある質問
├── data/                       # データセット格納用
├── models/                     # 学習済みモデル保存用
├── src/                        # ソースコード
│   ├── dataset.py              # データセット処理
│   ├── model.py                # モデル定義
│   ├── train.py                # 学習スクリプト
│   └── inference.py            # 推論スクリプト
└── requirements.txt            # 依存パッケージ
```

## 🛠️ セットアップ（初心者向け）

### ステップ1: Pythonのバージョン確認

```bash
python --version
```

Python 3.8以上が必要です。古い場合は[Python公式サイト](https://www.python.org/)からダウンロードしてください。

### ステップ2: 仮想環境の作成（推奨）

```bash
# 仮想環境を作成
python -m venv venv

# 仮想環境を有効化
source venv/bin/activate  # Mac/Linux
# または
venv\Scripts\activate     # Windows
```

💡 **仮想環境とは？** プロジェクトごとに独立したPython環境を作ることで、ライブラリの競合を防ぎます。

### ステップ3: 必要なライブラリのインストール

```bash
pip install -r requirements.txt
```

⏰ 所要時間: 5〜10分（インターネット速度によります）

## 💡 使い方（3つの方法）

### 方法1: 超簡単モード（初心者におすすめ）

```bash
python quickstart_simple.py
```

全て自動で設定されるので、コマンド1つで学習できます。

### 方法2: 設定ファイルを使う（少しカスタマイズしたい方）

```bash
python src/train.py --config config/beginner_config.yaml
```

`config/beginner_config.yaml`を編集することで、エポック数やバッチサイズなどを調整できます。

### 方法3: Pythonコードで細かく制御（上級者向け）

`notebooks/example_training.py`を参考に、自分でコードを書いてカスタマイズできます。

## 🔮 学習したモデルを使う

```bash
# インタラクティブモードで遊べる
python quickstart_predict.py

# または、コマンドラインで直接予測
python src/inference.py \
  --model_path ./models/my-first-model \
  --text "This movie is amazing!"
```

## 📊 学習の様子を確認する

別のターミナルで以下を実行：

```bash
tensorboard --logdir ./logs
```

ブラウザで http://localhost:6006 を開くと、学習の進捗がグラフで見られます。

## 🎓 サポートされているタスク

### 1. 感情分析・テキスト分類（初心者におすすめ）

**例**: 映画レビューがポジティブかネガティブか判定

```python
from src.model import ModelBuilder

# 2クラス分類（ポジティブ/ネガティブ）
model = ModelBuilder.build_sequence_classification_model(
    model_name="distilbert-base-uncased",  # 軽量で速い
    num_labels=2
)
```

**使える場面**: レビュー分析、スパム検出、カテゴリ分類

### 2. 固有表現認識（少し上級）

**例**: 「太郎さんは東京に住んでいる」→ 太郎（人名）、東京（地名）

```python
# 人名、地名、組織名などを抽出
model = ModelBuilder.build_token_classification_model(
    model_name="bert-base-uncased",
    num_labels=9  # BIOタグの数
)
```

**使える場面**: 文書からの情報抽出、エンティティ認識

### 3. 質問応答

**例**: 「太陽系で一番大きい惑星は？」→ 「木星」

```python
model = ModelBuilder.build_qa_model(
    model_name="bert-base-uncased"
)
```

**使える場面**: チャットボット、FAQ自動応答

### 4. テキスト生成（上級者向け）

**例**: 「昔々あるところに」→ 「昔々あるところにおじいさんとおばあさんが住んでいました...」

```python
model = ModelBuilder.build_causal_lm_model(
    model_name="gpt2"
)
```

**使える場面**: 文章の自動生成、コード補完

## 💾 自分のデータを使う

### CSVファイルを使う場合

**data/train.csv の例**:
```csv
text,label
"This is great!",1
"I don't like it",0
"Amazing product",1
```

**Pythonコード**:
```python
from src.dataset import CustomDatasetBuilder

dataset = CustomDatasetBuilder.from_csv(
    train_path="data/train.csv",
    validation_path="data/validation.csv",  # オプション
    test_path="data/test.csv"               # オプション
)
```

### JSONファイルを使う場合

**data/train.json の例**:
```json
[
  {"text": "This is great!", "label": 1},
  {"text": "I don't like it", "label": 0}
]
```

**Pythonコード**:
```python
dataset = CustomDatasetBuilder.from_json(
    train_path="data/train.json",
    test_path="data/test.json"
)
```

## 🌐 推奨モデル（初心者向け）

### 初めての方におすすめ

| モデル | 速度 | 精度 | 用途 |
|--------|------|------|------|
| `distilbert-base-uncased` | ⭐⭐⭐ 速い | ⭐⭐ 良い | 英語・初心者向け |
| `bert-base-uncased` | ⭐⭐ 普通 | ⭐⭐⭐ 高い | 英語・バランス型 |

### 日本語を扱う場合

| モデル | おすすめ度 | 特徴 |
|--------|----------|------|
| `cl-tohoku/bert-base-japanese` | ⭐⭐⭐ | 日本語で最もポピュラー |
| `bert-base-japanese` | ⭐⭐ | Google製の日本語BERT |

### 高精度を目指す場合（少し重い）

- `roberta-base`: BERTの改良版（英語）
- `roberta-large`: さらに高精度（要GPU）

## ⚠️ よくあるトラブルと解決法

### エラー: "CUDA out of memory"

**原因**: GPUメモリ不足

**解決法**:
```yaml
# config/beginner_config.yaml を編集
training:
  batch_size: 8 → 4  # 小さくする
  max_length: 256 → 128  # 短くする
```

### エラー: "No module named 'transformers'"

**原因**: ライブラリ未インストール

**解決法**:
```bash
pip install transformers
# または
pip install -r requirements.txt
```

### 学習が遅い / 止まっている

**CPUで実行している場合**:
- 正常です。GPUなら5分、CPUなら15分かかることも
- `quickstart_simple.py`は少量データなので短時間で完了します

**それでも遅い場合**:
```yaml
# データ量を減らす
dataset:
  name: "imdb"
# ↓ コード内で split を調整
split="train[:100]"  # 100件のみ使用
```

### その他の問題

**[よくある質問（FAQ）](docs/FAQ.md)** に20個以上のQ&Aがあります。

## 🎯 学習の流れ（初心者向け）

```
1. データ準備 📚
   ↓
2. モデル選択 🤖
   ↓
3. 学習実行 🏋️ (3〜5分)
   ↓
4. 評価 📊 (精度確認)
   ↓
5. 予測 🔮 (新しいデータで試す)
```

詳しくは **[基本概念ドキュメント](docs/CONCEPTS.md)** をご覧ください。

## 📖 さらに学ぶために

### 無料の学習リソース

1. **[Hugging Face コース](https://huggingface.co/learn/nlp-course/ja/chapter1/1)** - 日本語で学べる公式コース
2. **[Google Colab](https://colab.research.google.com/)** - 無料でGPUが使える環境
3. **[YouTube: Hugging Face チュートリアル](https://www.youtube.com/c/HuggingFace)** - 動画で学ぶ

### 公式ドキュメント

- [Transformers ドキュメント](https://huggingface.co/docs/transformers)
- [Datasets ドキュメント](https://huggingface.co/docs/datasets)
- [Trainer API リファレンス](https://huggingface.co/docs/transformers/main_classes/trainer)

## 🤝 コミュニティ

困ったときは気軽に質問してください：

- **Hugging Face フォーラム**: https://discuss.huggingface.co/
- **Stack Overflow**: `[huggingface]` タグで質問
- **Discord**: Hugging Face公式Discord

## 📝 ライセンス

このプロジェクトはMITライセンスの下で公開されています。

---

## 🎉 最後に

**AI学習は難しくありません！** まずは `python quickstart_simple.py` を実行して、AIモデルを作る楽しさを体験してください。

質問や改善案があれば、お気軽にお知らせください。

Happy Learning! 🚀
