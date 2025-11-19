"""
推論スクリプト

訓練済みモデルを使って座席推薦を実行します。
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path

from model import BaseSeatRecommender, SeatRecommendationSystem
from preprocessor import DataPreprocessor


def recommend_seats(
    user_id: int,
    model_path: str = "seat_recommendation/models/seat_recommender.pkl",
    preprocessor_path: str = "seat_recommendation/models/preprocessor.pkl",
    data_dir: str = "seat_recommendation/data",
    top_k: int = 5,
    verbose: bool = True
) -> pd.DataFrame:
    """
    特定ユーザーに座席を推薦

    Args:
        user_id: ユーザーID
        model_path: モデルファイルのパス
        preprocessor_path: 前処理器ファイルのパス
        data_dir: データディレクトリ
        top_k: 推薦する座席数
        verbose: 詳細表示

    Returns:
        推薦座席のDataFrame
    """
    if verbose:
        print(f"\n{'=' * 50}")
        print(f"座席推薦システム - ユーザー {user_id}")
        print(f"{'=' * 50}\n")

    # モデルと前処理器を読み込み
    if verbose:
        print("モデルを読み込んでいます...")

    recommender = BaseSeatRecommender.load(model_path)
    preprocessor = DataPreprocessor.load(preprocessor_path)

    if verbose:
        print("✓ モデル読み込み完了")

    # データを読み込み
    if verbose:
        print("データを読み込んでいます...")

    seat_info = pd.read_csv(f"{data_dir}/seat_info.csv")
    env_data = pd.read_csv(f"{data_dir}/environment_data.csv")
    user_profiles = pd.read_csv(f"{data_dir}/user_profiles.csv")

    if verbose:
        print("✓ データ読み込み完了")

    # ユーザー情報を確認
    user_data = user_profiles[user_profiles['user_id'] == user_id]
    if len(user_data) == 0:
        raise ValueError(f"ユーザーID {user_id} が見つかりません")

    if verbose:
        print(f"\nユーザー情報:")
        print(f"  タイプ: {user_data.iloc[0]['user_type']}")
        print(f"  好みの温度: {user_data.iloc[0]['preferred_temperature']:.1f}°C")
        print(f"  許容温度範囲: ±{user_data.iloc[0]['temperature_tolerance']:.1f}°C")
        print(f"  好みの照度: {user_data.iloc[0]['preferred_illuminance']:.0f} lux")
        print(f"  許容騒音レベル: {user_data.iloc[0]['max_acceptable_noise']:.1f} dB以下")
        print(f"  窓際の好み: {'はい' if user_data.iloc[0]['prefers_window'] else 'いいえ'}")
        print(f"  静かなエリアの好み: {'はい' if user_data.iloc[0]['prefers_quiet'] else 'いいえ'}")

    # 特徴量作成
    if verbose:
        print("\n特徴量を作成しています...")

    features_df = preprocessor.create_features(
        user_profiles=user_profiles,
        seat_info=seat_info,
        env_data=env_data,
        user_ratings=None
    )

    if verbose:
        print("✓ 特徴量作成完了")

    # 推薦システムを作成
    rec_system = SeatRecommendationSystem(recommender)

    # 座席を推薦
    if verbose:
        print(f"\nTop {top_k} 座席を推薦しています...")

    recommendations = rec_system.recommend_seats(
        user_id=user_id,
        features_df=features_df,
        top_k=top_k
    )

    if verbose:
        print(f"\n{'=' * 50}")
        print(f"推薦結果 (Top {top_k})")
        print(f"{'=' * 50}\n")

        for idx, row in recommendations.iterrows():
            print(f"【第{idx+1}位】 座席 {row['seat_id']} (行{row['seat_row']}, 列{row['seat_col']})")
            print(f"  予測評価スコア: {row['predicted_rating']:.2f} / 5.0")
            print(f"  平均温度: {row['seat_temp_mean']:.1f}°C")
            print(f"  平均騒音: {row['seat_noise_mean']:.1f} dB")
            print(f"  温度適合度: {row['temp_compatibility']:.2%}")
            print(f"  騒音適合度: {row['noise_compatibility']:.2%}")
            print(f"  照度適合度: {row['light_compatibility']:.2%}")
            print(f"  湿度適合度: {row['humidity_compatibility']:.2%}")
            print()

    return recommendations


def batch_recommend(
    user_ids: list,
    model_path: str = "seat_recommendation/models/seat_recommender.pkl",
    preprocessor_path: str = "seat_recommendation/models/preprocessor.pkl",
    data_dir: str = "seat_recommendation/data",
    top_k: int = 5,
    output_path: str = None
) -> dict:
    """
    複数ユーザーに座席を推薦

    Args:
        user_ids: ユーザーIDのリスト
        model_path: モデルファイルのパス
        preprocessor_path: 前処理器ファイルのパス
        data_dir: データディレクトリ
        top_k: 推薦する座席数
        output_path: 出力ファイルのパス

    Returns:
        推薦結果のディクショナリ
    """
    print(f"\n{'=' * 50}")
    print(f"バッチ推薦: {len(user_ids)} ユーザー")
    print(f"{'=' * 50}\n")

    # モデルと前処理器を読み込み
    recommender = BaseSeatRecommender.load(model_path)
    preprocessor = DataPreprocessor.load(preprocessor_path)

    # データを読み込み
    seat_info = pd.read_csv(f"{data_dir}/seat_info.csv")
    env_data = pd.read_csv(f"{data_dir}/environment_data.csv")
    user_profiles = pd.read_csv(f"{data_dir}/user_profiles.csv")

    # 特徴量作成
    features_df = preprocessor.create_features(
        user_profiles=user_profiles,
        seat_info=seat_info,
        env_data=env_data,
        user_ratings=None
    )

    # 推薦システムを作成
    rec_system = SeatRecommendationSystem(recommender)

    # 複数ユーザーに推薦
    recommendations = rec_system.recommend_for_multiple_users(
        user_ids=user_ids,
        features_df=features_df,
        top_k=top_k
    )

    print(f"✓ {len(recommendations)} ユーザーの推薦完了")

    # 結果を保存
    if output_path:
        import json

        # DataFrameをJSONに変換
        recommendations_json = {
            user_id: df.to_dict('records')
            for user_id, df in recommendations.items()
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(recommendations_json, f, indent=2, ensure_ascii=False)

        print(f"✓ 結果を保存: {output_path}")

    return recommendations


def main():
    parser = argparse.ArgumentParser(description='座席推薦システム - 推論')

    parser.add_argument('--user-id', type=int, help='ユーザーID (単一推薦)')
    parser.add_argument('--user-ids', type=int, nargs='+', help='ユーザーIDのリスト (バッチ推薦)')
    parser.add_argument('--model', type=str, default='seat_recommendation/models/seat_recommender.pkl',
                       help='モデルファイルのパス')
    parser.add_argument('--preprocessor', type=str, default='seat_recommendation/models/preprocessor.pkl',
                       help='前処理器ファイルのパス')
    parser.add_argument('--data-dir', type=str, default='seat_recommendation/data',
                       help='データディレクトリ')
    parser.add_argument('--top-k', type=int, default=5,
                       help='推薦する座席数')
    parser.add_argument('--output', type=str, help='出力ファイルのパス (バッチ推薦のみ)')

    args = parser.parse_args()

    if args.user_id is not None:
        # 単一ユーザー推薦
        recommend_seats(
            user_id=args.user_id,
            model_path=args.model,
            preprocessor_path=args.preprocessor,
            data_dir=args.data_dir,
            top_k=args.top_k
        )
    elif args.user_ids is not None:
        # バッチ推薦
        batch_recommend(
            user_ids=args.user_ids,
            model_path=args.model,
            preprocessor_path=args.preprocessor,
            data_dir=args.data_dir,
            top_k=args.top_k,
            output_path=args.output
        )
    else:
        parser.print_help()
        print("\nエラー: --user-id または --user-ids を指定してください")


if __name__ == "__main__":
    main()
