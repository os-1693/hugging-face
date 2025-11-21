# ❓ よくある質問（FAQ）

AI学習初心者の方からよく寄せられる質問と回答をまとめました。

## 📋 目次

1. [環境・インストール](#環境インストール)
2. [学習に関する質問](#学習に関する質問)
3. [エラー対処法](#エラー対処法)
4. [性能改善](#性能改善)
5. [実践的な質問](#実践的な質問)

---

## 環境・インストール

### Q1: Pythonのバージョンはどれが良い？

**A:** Python 3.8以上を推奨します。

```bash
# バージョン確認
python --version

# または
python3 --version
```

**推奨バージョン**: Python 3.9、3.10、3.11（PyTorch 2.0+の場合）

---

### Q2: GPUは必要？

**A:** 必須ではありませんが、あると便利です。

| 環境 | 学習時間（目安） | 推奨度 |
|------|----------------|--------|
| GPU（NVIDIA） | 3〜5分 | ⭐⭐⭐ 最適 |
| CPU | 10〜20分 | ⭐⭐ 可能 |

**初心者向けアドバイス**:
- まずはCPUで試してOK
- Google Colabなら無料でGPUが使える

---

### Q3: インストールでエラーが出る

**A:** 以下の順番で試してください:

```bash
# 1. pipをアップグレード
pip install --upgrade pip

# 2. requirements.txtを再インストール
pip install -r requirements.txt

# 3. それでもダメなら個別インストール
pip install torch transformers datasets evaluate
```

**よくあるエラー**:
```
ERROR: Could not find a version that satisfies the requirement...
```
→ Pythonのバージョンが古い可能性があります

---

### Q4: どれくらいのメモリ（RAM）が必要？

**A:** 最低8GB、推奨16GB以上

| メモリ | できること |
|-------|-----------|
| 4GB | ❌ 厳しい |
| 8GB | ⚠️ 小さいモデルなら可能 |
| 16GB | ✓ 快適 |
| 32GB以上 | ✓ 大きいモデルも可能 |

**メモリ節約のコツ**:
```yaml
# config/beginner_config.yaml
training:
  batch_size: 4  # 小さくする
  max_length: 128  # 短くする
```

---

## 学習に関する質問

### Q5: 学習にどれくらい時間がかかる？

**A:** 環境とデータ量によります

**quickstart_simple.py の場合**:
- GPU: 3〜5分
- CPU: 10〜20分

**フルデータセットの場合**:
- GPU: 30分〜数時間
- CPU: 数時間〜1日

---

### Q6: どれくらいの精度が「良い」の？

**A:** タスクによりますが、一般的な目安:

| 精度 | 評価 | コメント |
|------|------|----------|
| 90%以上 | 優秀 | 実用レベル |
| 80〜90% | 良い | 十分使える |
| 70〜80% | まずまず | 改善の余地あり |
| 70%未満 | 要改善 | パラメータ調整が必要 |

**注意**: ベースライン（ランダム予測）と比較することが重要
- 2クラス分類: 50%（コインを投げるのと同じ）
- 10クラス分類: 10%

---

### Q7: エポック数はいくつが良い？

**A:** 3〜5回が初心者におすすめ

```yaml
num_epochs: 3  # 初めての場合
num_epochs: 5  # もっと精度を上げたい場合
num_epochs: 10 # 慎重に（過学習のリスク）
```

**判断基準**:
```
エポック1: Accuracy 75%
エポック2: Accuracy 82%
エポック3: Accuracy 85%
エポック4: Accuracy 86%
エポック5: Accuracy 85% ← 下がった！（過学習の兆候）
```

→ エポック4で止めるのが良い

---

### Q8: 学習中に何をすれば良い？

**A:** TensorBoardで進捗を確認しましょう

```bash
# 別のターミナルで実行
tensorboard --logdir ./logs
```

ブラウザで http://localhost:6006 を開く

**見るべきグラフ**:
- **Loss（損失）**: 下がっていればOK
- **Accuracy（精度）**: 上がっていればOK

---

## エラー対処法

### Q9: "CUDA out of memory" エラー

**A:** GPUメモリ不足です。以下を試してください:

**方法1: バッチサイズを減らす**
```yaml
batch_size: 16 → 8
batch_size: 8 → 4
```

**方法2: モデルを小さくする**
```yaml
model:
  name: "bert-base-uncased" → "distilbert-base-uncased"
```

**方法3: シーケンス長を短くする**
```yaml
max_length: 512 → 256
```

---

### Q10: "Connection Error" / ダウンロードエラー

**A:** ネットワークの問題です

**対処法1: リトライ**
```bash
# もう一度実行するだけ（自動でリトライされます）
python quickstart_simple.py
```

**対処法2: プロキシ設定**
```bash
# 会社や学校のネットワークの場合
export HTTP_PROXY=http://proxy.example.com:8080
export HTTPS_PROXY=http://proxy.example.com:8080
```

**対処法3: オフラインモード**
```python
# 一度ダウンロードしたモデルはキャッシュされます
# ~/.cache/huggingface/ に保存されています
```

---

### Q11: "No module named 'XXX'" エラー

**A:** ライブラリがインストールされていません

```bash
# 個別にインストール
pip install XXX

# 例
pip install transformers
pip install torch
pip install datasets
```

---

### Q12: 学習が途中で止まる

**A:** いくつかの原因が考えられます

**原因1: メモリ不足**
```
→ バッチサイズを減らす
→ 不要なアプリを閉じる
```

**原因2: ディスク容量不足**
```bash
# 空き容量を確認
df -h

# キャッシュを削除
rm -rf ~/.cache/huggingface/datasets/*
```

**原因3: 長時間実行で接続切れ**
```
→ nohup や screen を使う
→ または短いエポック数で試す
```

---

## 性能改善

### Q13: 精度を上げるには？

**A:** 以下の方法を試してください

**方法1: エポック数を増やす**
```yaml
num_epochs: 3 → 5
```

**方法2: データ量を増やす**
```python
# 1000件 → 5000件
split="train[:5000]"
```

**方法3: より良いモデルを使う**
```yaml
model:
  name: "distilbert-base-uncased" → "roberta-base"
```

**方法4: 学習率を調整**
```yaml
learning_rate: 2e-5 → 3e-5
```

---

### Q14: 学習を速くするには？

**A:** いくつかの高速化テクニック

**方法1: GPUを使う**
```
CPUの3〜5倍速い
```

**方法2: 軽量モデルを使う**
```yaml
model:
  name: "distilbert-base-uncased"  # BERTの60%の速度
```

**方法3: データ量を減らす（テスト時）**
```python
split="train[:1000]"  # まず小規模で試す
```

**方法4: 混合精度学習**
```yaml
training:
  fp16: true  # GPUが必要
```

---

### Q15: 過学習を防ぐには？

**A:** 以下の対策が有効です

**対策1: 早期停止を使う**
```yaml
training:
  early_stopping: true
  early_stopping_patience: 3
```

**対策2: ドロップアウトを追加**
```yaml
model:
  config_overrides:
    dropout: 0.1
    attention_dropout: 0.1
```

**対策3: データ拡張**
```python
# より多くの訓練データを用意
```

**対策4: 正則化を強化**
```yaml
training:
  weight_decay: 0.01 → 0.05
```

---

## 実践的な質問

### Q16: 日本語テキストを分析したい

**A:** 日本語モデルを使いましょう

**推奨モデル一覧**:

| モデル | 特徴 | 用途 |
|--------|------|------|
| `cl-tohoku/bert-base-japanese-v3` | 高精度、最新版 | 汎用 |
| `cl-tohoku/bert-base-japanese-char-v3` | 文字ベース | 未知語に強い |
| `studio-ousia/luke-japanese-base-lite` | エンティティ認識に強い | NER |
| `rinna/japanese-roberta-base` | RoBERTaベース | 汎用 |

**設定例**:
```yaml
model:
  name: "cl-tohoku/bert-base-japanese-v3"
  num_labels: 2
```

**コード例**:
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "cl-tohoku/bert-base-japanese-v3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 日本語テキストで予測
text = "この映画は素晴らしかった"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
```

**日本語データセット例**:
- `llm-book/livedoor-news-corpus`: ニュース分類
- `shunk031/wrime`: 感情分析
- `cl-nagoya/ner-wikipedia-dataset`: 固有表現認識

**注意点**:
- 日本語モデルはトークナイザーが自動で日本語対応
- MeCab/Sudachiなどの形態素解析器が必要な場合あり
- `pip install fugashi unidic-lite` で対応可能

---

### Q17: 自分のデータで学習したい

**A:** CSVまたはJSONファイルを用意してください

**CSVの例**:
```csv
text,label
"素晴らしい商品です",1
"期待外れでした",0
```

**使い方**:
```python
from src.dataset import CustomDatasetBuilder

dataset = CustomDatasetBuilder.from_csv(
    train_path="data/train.csv",
    test_path="data/test.csv"
)
```

---

### Q18: 学習済みモデルを配布したい

**A:** Hugging Face Hubにアップロードできます

```bash
# Hugging Faceにログイン
huggingface-cli login

# モデルをアップロード
python -c "
from transformers import AutoModel, AutoTokenizer
model = AutoModel.from_pretrained('./models/my-first-model')
tokenizer = AutoTokenizer.from_pretrained('./models/my-first-model')
model.push_to_hub('your-username/model-name')
tokenizer.push_to_hub('your-username/model-name')
"
```

---

### Q19: Google Colabで実行したい

**A:** 以下の手順で簡単にできます

```python
# Colab上で実行
!git clone https://github.com/your-repo/hugging-face.git
%cd hugging-face
!pip install -r requirements.txt
!python quickstart_simple.py
```

**メリット**:
- 無料でGPUが使える
- 環境構築不要
- ブラウザだけでOK

---

### Q20: 商用利用できる？

**A:** モデルのライセンスによります

**一般的なライセンス**:
- **MIT**: 商用利用OK
- **Apache 2.0**: 商用利用OK
- **GPL**: 要注意（ソースコード公開が必要な場合あり）

**確認方法**:
```python
# モデルページで確認
# https://huggingface.co/bert-base-uncased
# → "Model card" タブ → "License" セクション
```

---

## 🆘 それでも解決しない場合

1. **GitHub Issues**: プロジェクトのIssuesページで質問
2. **Stack Overflow**: `[huggingface] [transformers]` タグで質問
3. **Hugging Face フォーラム**: https://discuss.huggingface.co/

---

## 📚 さらに学ぶには

- [Hugging Face Course（無料）](https://huggingface.co/learn/nlp-course)
- [TensorFlow チュートリアル](https://www.tensorflow.org/tutorials)
- [PyTorch チュートリアル](https://pytorch.org/tutorials/)

---

**このFAQに追加してほしい質問があれば、お気軽にお知らせください！**
