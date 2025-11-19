"""
座席推薦システム - デモスクリプト

初心者向けの簡単なデモ実行スクリプトです。
このスクリプトを実行するだけで、データ生成からモデル訓練、推薦までの全フローを体験できます。
"""

import os
import sys

# srcディレクトリをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_generator import generate_all_data
from preprocessor import DataPreprocessor, split_data
from model import create_recommender, SeatRecommendationSystem
from inference import recommend_seats
import pandas as pd


def print_header(text):
    """ヘッダーを表示"""
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60 + "\n")


def main():
    print_header("座席推薦システム - デモ")
    print("このデモでは、以下の流れを体験できます:")
    print("1. サンプルデータの生成")
    print("2. 機械学習モデルの訓練")
    print("3. 座席の推薦")
    print()
    input("Enterキーを押して開始してください...")

    # ステップ1: データ生成
    print_header("ステップ1: サンプルデータの生成")
    print("室内環境データ、座席情報、ユーザーアンケートデータを生成します。")
    print()

    data_dir = "seat_recommendation/data"
    os.makedirs(data_dir, exist_ok=True)

    # 小規模なデータセットで高速実行
    data = generate_all_data(
        num_seats=30,          # 30座席
        num_users=100,         # 100ユーザー
        num_timestamps=500,    # 500タイムスタンプ
        num_ratings=2000,      # 2000評価
        output_dir=data_dir
    )

    print("\nデータ生成完了！")
    input("\nEnterキーを押して次へ...")

    # ステップ2: 特徴量作成と前処理
    print_header("ステップ2: 特徴量作成と前処理")
    print("機械学習モデルで使用する特徴量を作成します。")
    print()

    preprocessor = DataPreprocessor()
    features_df = preprocessor.create_features(
        user_profiles=data['user_profiles'],
        seat_info=data['seat_info'],
        env_data=data['environment_data'],
        user_ratings=data['user_ratings']
    )

    print(f"✓ 特徴量データ作成完了: {features_df.shape}")
    print(f"✓ 特徴量数: {len([col for col in features_df.columns if col not in ['user_id', 'seat_id', 'overall_rating']])}")

    # データ分割
    X, y, feature_names = preprocessor.prepare_training_data(features_df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    print(f"\n✓ 訓練データ: {X_train.shape[0]} サンプル")
    print(f"✓ 検証データ: {X_val.shape[0]} サンプル")
    print(f"✓ テストデータ: {X_test.shape[0]} サンプル")

    input("\nEnterキーを押して次へ...")

    # ステップ3: モデル訓練
    print_header("ステップ3: 機械学習モデルの訓練")
    print("ランダムフォレストモデルを使って座席推薦モデルを訓練します。")
    print()

    # ランダムフォレストモデル（高速で精度が良い）
    recommender = create_recommender(
        'random_forest',
        n_estimators=50,      # 木の数（デモなので少なめ）
        max_depth=10,         # 木の深さ
        random_state=42
    )

    print("訓練中...")
    metrics = recommender.train(X_train, y_train, X_val, y_val, feature_names)

    print("\n訓練完了！")
    print(f"\n訓練データ性能:")
    print(f"  MSE: {metrics['train_mse']:.4f}")
    print(f"  MAE: {metrics['train_mae']:.4f}")
    print(f"  R²スコア: {metrics['train_r2']:.4f}")

    print(f"\n検証データ性能:")
    print(f"  MSE: {metrics['val_mse']:.4f}")
    print(f"  MAE: {metrics['val_mae']:.4f}")
    print(f"  R²スコア: {metrics['val_r2']:.4f}")

    # 特徴量重要度
    importance_df = recommender.get_feature_importance()
    if importance_df is not None:
        print("\n重要な特徴量 Top 5:")
        for idx, row in importance_df.head(5).iterrows():
            print(f"  {idx+1}. {row['feature']}: {row['importance']:.4f}")

    input("\nEnterキーを押して次へ...")

    # ステップ4: モデル保存
    print_header("ステップ4: モデル保存")

    model_dir = "seat_recommendation/models"
    os.makedirs(model_dir, exist_ok=True)

    model_path = f"{model_dir}/seat_recommender.pkl"
    preprocessor_path = f"{model_dir}/preprocessor.pkl"

    recommender.save(model_path)
    preprocessor.save(preprocessor_path)

    print(f"✓ モデル保存: {model_path}")
    print(f"✓ 前処理器保存: {preprocessor_path}")

    input("\nEnterキーを押して次へ...")

    # ステップ5: 座席推薦デモ
    print_header("ステップ5: 座席推薦デモ")
    print("いくつかのユーザーに対して座席を推薦してみます。")
    print()

    # ランダムに3人のユーザーを選択
    import random
    sample_users = random.sample(range(100), 3)

    rec_system = SeatRecommendationSystem(recommender)

    for user_id in sample_users:
        user_info = data['user_profiles'][data['user_profiles']['user_id'] == user_id].iloc[0]

        print(f"\n{'─' * 60}")
        print(f"ユーザー {user_id} の情報:")
        print(f"  タイプ: {user_info['user_type']}")
        print(f"  好みの温度: {user_info['preferred_temperature']:.1f}°C (±{user_info['temperature_tolerance']:.1f}°C)")
        print(f"  好みの照度: {user_info['preferred_illuminance']:.0f} lux")
        print(f"  許容騒音レベル: {user_info['max_acceptable_noise']:.1f} dB以下")
        print(f"  窓際の好み: {'はい' if user_info['prefers_window'] else 'いいえ'}")

        # 座席推薦
        recommendations = rec_system.recommend_seats(
            user_id=user_id,
            features_df=features_df,
            top_k=3
        )

        print(f"\n推薦座席 Top 3:")
        for idx, row in recommendations.iterrows():
            print(f"  {idx+1}位: 座席 {row['seat_id']} (行{row['seat_row']}, 列{row['seat_col']})")
            print(f"       予測評価: {row['predicted_rating']:.2f}/5.0")
            print(f"       温度: {row['seat_temp_mean']:.1f}°C, 騒音: {row['seat_noise_mean']:.1f}dB")
            print(f"       適合度 - 温度:{row['temp_compatibility']:.0%}, 騒音:{row['noise_compatibility']:.0%}")

        if user_id != sample_users[-1]:
            input("\nEnterキーを押して次のユーザーへ...")

    # まとめ
    print_header("デモ完了！")
    print("座席推薦システムのデモを体験していただきありがとうございました。")
    print()
    print("次のステップ:")
    print("  1. src/train.py を使ってより大規模なデータで訓練")
    print("  2. src/inference.py を使って特定ユーザーに推薦")
    print("  3. パラメータを調整してモデル性能を改善")
    print()
    print("詳しくは README.md をご覧ください。")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nデモを中断しました。")
    except Exception as e:
        print(f"\n\nエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
