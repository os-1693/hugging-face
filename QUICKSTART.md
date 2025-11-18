# 🚀 クイックスタートガイド - AI学習初心者向け

このガイドでは、初めてAIモデルを学習する方でも簡単に始められるように、ステップバイステップで説明します。

## 📋 目次

1. [環境のセットアップ](#1-環境のセットアップ)
2. [最初のモデル学習（5分で完了）](#2-最初のモデル学習5分で完了)
3. [学習結果の確認](#3-学習結果の確認)
4. [モデルの使い方](#4-モデルの使い方)
5. [次のステップ](#5-次のステップ)

---

## 1. 環境のセットアップ

### ステップ 1: Pythonのインストール確認

ターミナル（コマンドプロンプト）を開いて、以下のコマンドを実行してください：

```bash
python --version
```

Python 3.8以上がインストールされていることを確認してください。

### ステップ 2: プロジェクトのディレクトリに移動

```bash
cd hugging-face
```

### ステップ 3: 必要なライブラリのインストール

```bash
pip install -r requirements.txt
```

⏰ **所要時間**: 約5〜10分（インターネット速度によります）

💡 **注意**: エラーが出た場合は、`pip install --upgrade pip` を先に実行してみてください。

---

## 2. 最初のモデル学習（5分で完了）

### オプションA: 超簡単モード（推奨）

以下のコマンド1つで学習が開始されます：

```bash
python quickstart_simple.py
```

このスクリプトは：
- ✅ 小さなデータセットを自動ダウンロード
- ✅ AIモデルを自動設定
- ✅ 学習を実行（約3〜5分）
- ✅ 結果を表示

### オプションB: 設定ファイルを使う方法

```bash
python src/train.py --config config/beginner_config.yaml
```

---

## 3. 学習結果の確認

### 学習が完了すると…

```
✓ Training completed!
✓ Model saved to: ./models/my-first-model
✓ Accuracy: 85.5%
```

このように表示されれば成功です！

### 学習の様子を可視化する

別のターミナルで以下を実行：

```bash
tensorboard --logdir ./logs
```

ブラウザで `http://localhost:6006` を開くと、学習の進捗がグラフで見られます。

---

## 4. モデルの使い方

### 学習したモデルで予測してみよう

```bash
python quickstart_predict.py
```

または、自分のテキストで試す：

```bash
python src/inference.py \
  --model_path ./models/my-first-model \
  --text "This movie is amazing!"
```

結果の見方：
```
Predicted class: 1 (ポジティブ)
Confidence: 0.9234 (92.34%の確信度)
```

---

## 5. 次のステップ

### 🎯 やってみよう

1. **別のデータセットを試す**
   - `config/beginner_config.yaml`の`dataset.name`を変更
   - 例: `"imdb"` → `"sst2"` に変更

2. **学習時間を調整する**
   - `num_epochs: 3` → `num_epochs: 5` に増やす
   - より良い精度が得られるかも！

3. **日本語モデルを試す**
   - `model.name: "bert-base-uncased"` → `"cl-tohoku/bert-base-japanese"`

### 📚 学習リソース

- [基本概念の説明](docs/CONCEPTS.md) - AIモデル学習の基礎を理解
- [よくある質問](docs/FAQ.md) - トラブルシューティング
- [サンプル集](examples/) - いろいろなタスクの例

---

## ❓ 困ったときは

### よくあるエラーと対処法

**エラー: "No module named 'transformers'"**
```bash
pip install transformers
```

**エラー: "CUDA out of memory"**
- `config/beginner_config.yaml`で`batch_size: 8`に減らす

**学習が遅い**
- CPUで実行している場合は通常より時間がかかります（正常です）
- GPUがあれば自動的に使用されます

---

## 🎉 おめでとうございます！

初めてのAIモデル学習が完了しました。
これで基本的な流れが理解できたはずです。

次は実際のプロジェクトに挑戦してみましょう！
