# 座席推薦システム - 機械学習プロジェクト

室内環境データとユーザーアンケートデータをもとに、最適な座席を推薦する機械学習システムです。

## プロジェクト概要

このプロジェクトは、以下のデータを活用して座席を推薦します：

### 入力データ
1. **室内環境データ（多次元）**
   - 温度 (°C)
   - 湿度 (%)
   - 照度 (lux)
   - 騒音レベル (dB)
   - CO2濃度 (ppm)

2. **ユーザーアンケートデータ**
   - 好みの温度と許容範囲
   - 好みの照度と許容範囲
   - 騒音感度
   - 窓際の好み
   - 静かなエリアの好み
   - モニターの必要性
   - スタンディングデスクの好み

3. **座席情報**
   - 位置（行・列）
   - 窓際かどうか
   - 空調の近くか
   - 出入り口の近くか
   - 静かなエリアか
   - 設備（モニター、スタンディングデスク）

### 出力
- 各ユーザーに対する推薦座席のランキング
- 予測される満足度スコア
- 環境適合度の詳細

## 5分でスタート（超簡単モード）

### 方法1: uvを使う（推奨・高速）

[uv](https://github.com/astral-sh/uv)は、Rustで書かれた超高速なPythonパッケージマネージャーです。

#### uvのインストール

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# pipでインストール
pip install uv
```

#### プロジェクトのセットアップと実行

```bash
cd seat_recommendation

# 依存パッケージをインストール（pipの10-100倍速い！）
uv pip install -e .

# または、requirements.txtから
uv pip install -r requirements.txt

# デモを実行
python demo.py
```

### 方法2: 従来のpipを使う

```bash
cd seat_recommendation

# 仮想環境を作成（推奨）
python -m venv venv
source venv/bin/activate  # Mac/Linux
# または venv\Scripts\activate  # Windows

# 必要なライブラリをインストール
pip install -r requirements.txt

# デモを実行
python demo.py
```

### デモで体験できること

以下の全てを自動で実行します：
- サンプルデータの自動生成
- 機械学習モデルの訓練
- 座席の推薦

所要時間: 3〜5分

## プロジェクト構成

```
seat_recommendation/
├── README.md                    # このファイル
├── requirements.txt             # 依存パッケージ
├── demo.py                      # デモスクリプト（初心者向け）
├── config/
│   └── config.yaml              # 設定ファイル
├── data/                        # データ保存ディレクトリ
│   ├── seat_info.csv            # 座席情報（生成後）
│   ├── environment_data.csv     # 環境データ（生成後）
│   ├── user_profiles.csv        # ユーザープロファイル（生成後）
│   └── user_ratings.csv         # 評価データ（生成後）
├── src/
│   ├── __init__.py
│   ├── data_generator.py        # データ生成
│   ├── preprocessor.py          # データ前処理
│   ├── model.py                 # 機械学習モデル
│   ├── train.py                 # 訓練スクリプト
│   └── inference.py             # 推論スクリプト
├── models/                      # 訓練済みモデル保存先
├── output/                      # 実験結果保存先
└── notebooks/                   # Jupyterノートブック用
```

## 詳細な使い方

### ステップ1: データ生成

サンプルデータを生成します：

```bash
cd seat_recommendation
python -c "from src.data_generator import generate_all_data; generate_all_data()"
```

**カスタマイズ例**：
```python
from src.data_generator import generate_all_data

generate_all_data(
    num_seats=100,        # 座席数を増やす
    num_users=500,        # ユーザー数を増やす
    num_timestamps=2000,  # より多くの環境データ
    num_ratings=10000,    # より多くの評価データ
    output_dir="data"
)
```

### ステップ2: モデル訓練

#### 方法1: 設定ファイルを使う（推奨）

```bash
python src/train.py --config config/config.yaml
```

設定ファイル (`config/config.yaml`) を編集してパラメータを調整できます。

#### 方法2: コマンドライン引数を使う

```bash
python src/train.py --model-type random_forest --num-seats 50 --num-users 200
```

#### 方法3: Pythonコードで実行

```python
from src.train import train_model

recommender, preprocessor, features_df, results = train_model(
    config_path='config/config.yaml'
)

print(f"テスト R²スコア: {results['metrics']['test_r2']:.4f}")
```

### ステップ3: 座席推薦

#### 単一ユーザーへの推薦

```bash
python src/inference.py --user-id 0 --top-k 5
```

#### 複数ユーザーへの推薦

```bash
python src/inference.py --user-ids 0 1 2 3 4 --output output/recommendations.json
```

#### Pythonコードで推薦

```python
from src.inference import recommend_seats

recommendations = recommend_seats(
    user_id=0,
    model_path="models/seat_recommender.pkl",
    preprocessor_path="models/preprocessor.pkl",
    data_dir="data",
    top_k=5
)

print(recommendations)
```

## サポートされているモデル

### 1. ランダムフォレスト（推奨）

**特徴**: バランスが良く、特徴量の重要度が分かる

```yaml
model:
  type: random_forest
  params:
    n_estimators: 100      # 木の数
    max_depth: 10          # 木の深さ
    random_state: 42
```

**長所**:
- 精度が高い
- 過学習しにくい
- 特徴量の重要度が分かる
- 訓練が速い

### 2. 勾配ブースティング

**特徴**: 最高精度を目指す場合

```yaml
model:
  type: gradient_boosting
  params:
    n_estimators: 100
    learning_rate: 0.1
    max_depth: 3
    random_state: 42
```

**長所**:
- 精度が非常に高い
- 複雑なパターンを学習

**短所**:
- 訓練に時間がかかる
- 過学習しやすい

### 3. ニューラルネットワーク

**特徴**: 大規模データで威力を発揮

```yaml
model:
  type: neural_network
  params:
    hidden_layer_sizes: [100, 50]
    activation: relu
    learning_rate_init: 0.001
    max_iter: 200
    random_state: 42
```

**長所**:
- 非線形パターンを学習
- 大規模データで高精度

**短所**:
- 訓練に時間がかかる
- ハイパーパラメータ調整が重要

### 4. 線形回帰

**特徴**: シンプルで解釈しやすい

```yaml
model:
  type: linear
  params:
    alpha: 1.0  # 正則化パラメータ
```

**長所**:
- 訓練が非常に速い
- 解釈しやすい

**短所**:
- 精度は低め
- 複雑なパターンを学習できない

## データの詳細

### 室内環境データの特性

| 指標 | 範囲 | 単位 | 説明 |
|------|------|------|------|
| 温度 | 20-28 | °C | 座席ごとに異なる（窓際、空調の影響） |
| 湿度 | 30-70 | % | 時間帯による変動あり |
| 照度 | 200-800 | lux | 窓際は明るい、日中の変動大 |
| 騒音 | 30-70 | dB | 出入り口付近は騒がしい |
| CO2 | 400-1200 | ppm | 人の多さに依存 |

### ユーザータイプ

システムは以下のユーザータイプを認識します：

1. **cold_sensitive（寒がり）**: 高めの温度を好む
2. **heat_sensitive（暑がり）**: 低めの温度を好む
3. **noise_sensitive（騒音に敏感）**: 静かな環境を好む
4. **light_sensitive（光に敏感）**: 暗めの環境を好む
5. **balanced（バランス型）**: 標準的な環境を好む

## 評価メトリクス

モデルの性能は以下の指標で評価されます：

- **MSE（平均二乗誤差）**: 予測の誤差の二乗平均（低いほど良い）
- **MAE（平均絶対誤差）**: 予測の誤差の絶対値平均（低いほど良い）
- **RMSE（二乗平均平方根誤差）**: MSEの平方根（低いほど良い）
- **R²スコア**: 決定係数（1に近いほど良い、1が最高）

**良いモデルの目安**:
- R²スコア > 0.7: 良好
- R²スコア > 0.8: 優秀
- R²スコア > 0.9: 非常に優秀

## カスタマイズガイド

### 1. 独自のデータを使う

`data/` ディレクトリに以下のCSVファイルを配置：

**seat_info.csv**:
```csv
seat_id,row,col,is_window,near_ac,near_entrance,quiet_area,has_monitor,has_standing_desk
0,0,0,1,1,0,1,1,0
1,0,1,0,1,0,1,1,0
...
```

**user_profiles.csv**:
```csv
user_id,user_type,preferred_temperature,temperature_tolerance,preferred_humidity,...
0,cold_sensitive,25.5,1.0,50.0,...
1,heat_sensitive,21.0,0.8,48.0,...
...
```

### 2. 新しい特徴量を追加

`src/preprocessor.py` の `create_features` メソッドを編集：

```python
# 新しい特徴量を追加
feature_dict['new_feature'] = calculate_new_feature(user, seat)
```

### 3. ハイパーパラメータ調整

`config/config.yaml` を編集してモデルのパラメータを調整：

```yaml
model:
  params:
    n_estimators: 200      # 木の数を増やす
    max_depth: 15          # 深さを増やす
```

## よくある質問

### Q1: データ生成にどのくらい時間がかかりますか？

A: 標準設定（50座席、200ユーザー）で約10秒です。

### Q2: 訓練にどのくらい時間がかかりますか？

A: ランダムフォレストで約30秒〜1分です。データサイズやモデルによって変わります。

### Q3: GPUは必要ですか？

A: いいえ。このプロジェクトはCPUのみで十分高速に動作します。

### Q4: 実データで使うにはどうすれば？

A: `data/` ディレクトリに実際のCSVファイルを配置し、データ生成をスキップしてください。

### Q5: モデルの精度を上げるには？

A: 以下を試してください：
1. データ量を増やす（num_users, num_ratingsを増やす）
2. モデルを変更（gradient_boostingを試す）
3. ハイパーパラメータを調整
4. 新しい特徴量を追加

## 技術スタック

- **Python 3.8+**
- **pandas**: データ処理
- **numpy**: 数値計算
- **scikit-learn**: 機械学習
- **joblib**: モデルの保存・読み込み（安全性向上）
- **PyYAML**: 設定ファイル
- **uv**: 高速パッケージマネージャー（オプション）

## 学習の流れ

```
1. データ準備
   ├─ 室内環境データ
   ├─ ユーザープロファイル
   ├─ 座席情報
   └─ 過去の評価データ

2. 特徴量エンジニアリング
   ├─ ユーザーの好みと座席特性の適合度計算
   ├─ 環境データの集約（平均、標準偏差）
   └─ カテゴリ特徴量の一致度

3. モデル訓練
   ├─ データ分割（訓練/検証/テスト）
   ├─ モデル学習
   └─ ハイパーパラメータ調整

4. 評価
   ├─ テストデータで性能評価
   └─ 特徴量重要度の分析

5. 推論
   └─ ユーザーごとに最適座席を推薦
```

## 実用例

### オフィス環境での利用

```python
# 新入社員に座席を割り当て
new_employee_id = 150
recommendations = recommend_seats(user_id=new_employee_id, top_k=3)
print(f"新入社員におすすめの座席: {recommendations['seat_id'].tolist()}")
```

### 座席配置の最適化

```python
# 全ユーザーに推薦して、満足度を最大化する配置を計算
from src.inference import batch_recommend

all_user_ids = list(range(200))
all_recommendations = batch_recommend(
    user_ids=all_user_ids,
    top_k=1,
    output_path="output/optimal_assignment.json"
)
```

### A/Bテスト

```python
# モデルAとモデルBの性能を比較
model_a = create_recommender('random_forest', n_estimators=50)
model_b = create_recommender('gradient_boosting', n_estimators=100)

# 両方訓練して比較...
```

## 今後の拡張アイデア

1. **時系列予測**: 時間帯ごとの座席推薦
2. **協調フィルタリング**: 類似ユーザーの好みを活用
3. **Deep Learning**: より複雑なパターンの学習
4. **リアルタイム推薦**: センサーデータと連携
5. **多目的最適化**: 個人の満足度とチーム配置の両立

## ライセンス

MIT License

## サポート

問題が発生した場合は、以下を確認してください：

1. Pythonバージョン: 3.8以上
2. 依存パッケージ: `pip install -r requirements.txt`
3. データファイル: `data/` ディレクトリに必要なCSVがあるか

---

**Happy Learning!**

座席推薦システムの構築を楽しんでください。質問や改善案があれば、お気軽にお知らせください。
