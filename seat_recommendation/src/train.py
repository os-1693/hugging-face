"""
モデル訓練スクリプト

座席推薦モデルの訓練を実行します。
"""

import os
import sys
import argparse
import yaml
import pandas as pd
import numpy as np
import json
from pathlib import Path

from data_generator import generate_all_data
from preprocessor import DataPreprocessor, split_data
from model import create_recommender, SeatRecommendationSystem


def train_model(config_path: str = None, **kwargs):
    """
    モデルを訓練

    Args:
        config_path: 設定ファイルのパス
        **kwargs: コマンドライン引数で上書きするパラメータ
    """
    # デフォルト設定
    config = {
        'data': {
            'num_seats': 50,
            'num_users': 200,
            'num_timestamps': 1000,
            'num_ratings': 5000,
            'data_dir': 'seat_recommendation/data',
            'generate_new': False
        },
        'model': {
            'type': 'random_forest',
            'params': {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42
            }
        },
        'training': {
            'train_ratio': 0.7,
            'val_ratio': 0.15,
            'random_seed': 42
        },
        'output': {
            'model_dir': 'seat_recommendation/models',
            'output_dir': 'seat_recommendation/output'
        }
    }

    # 設定ファイルを読み込み
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            file_config = yaml.safe_load(f)
            # 設定をマージ
            for key in file_config:
                if key in config:
                    config[key].update(file_config[key])

    # コマンドライン引数で上書き
    for key, value in kwargs.items():
        if value is not None:
            # ネストした設定を処理
            if '.' in key:
                parts = key.split('.')
                if parts[0] in config:
                    config[parts[0]][parts[1]] = value
            else:
                config[key] = value

    print("=" * 50)
    print("座席推薦モデル訓練")
    print("=" * 50)
    print(f"\n設定:")
    print(json.dumps(config, indent=2, ensure_ascii=False))

    # ディレクトリ作成
    os.makedirs(config['data']['data_dir'], exist_ok=True)
    os.makedirs(config['output']['model_dir'], exist_ok=True)
    os.makedirs(config['output']['output_dir'], exist_ok=True)

    # ステップ1: データの準備
    print("\n" + "=" * 50)
    print("ステップ1: データの準備")
    print("=" * 50)

    data_dir = config['data']['data_dir']
    seat_info_path = f"{data_dir}/seat_info.csv"

    if config['data']['generate_new'] or not os.path.exists(seat_info_path):
        print("新しいデータを生成しています...")
        data = generate_all_data(
            num_seats=config['data']['num_seats'],
            num_users=config['data']['num_users'],
            num_timestamps=config['data']['num_timestamps'],
            num_ratings=config['data']['num_ratings'],
            output_dir=data_dir
        )
    else:
        print("既存のデータを読み込んでいます...")
        data = {
            'seat_info': pd.read_csv(f"{data_dir}/seat_info.csv"),
            'environment_data': pd.read_csv(f"{data_dir}/environment_data.csv"),
            'user_profiles': pd.read_csv(f"{data_dir}/user_profiles.csv"),
            'user_ratings': pd.read_csv(f"{data_dir}/user_ratings.csv")
        }
        print(f"✓ 座席数: {len(data['seat_info'])}")
        print(f"✓ ユーザー数: {len(data['user_profiles'])}")
        print(f"✓ 評価数: {len(data['user_ratings'])}")

    # ステップ2: 特徴量作成
    print("\n" + "=" * 50)
    print("ステップ2: 特徴量作成")
    print("=" * 50)

    preprocessor = DataPreprocessor()
    features_df = preprocessor.create_features(
        user_profiles=data['user_profiles'],
        seat_info=data['seat_info'],
        env_data=data['environment_data'],
        user_ratings=data['user_ratings']
    )

    print(f"✓ 特徴量データ作成完了: {features_df.shape}")
    print(f"✓ 特徴量数: {len([col for col in features_df.columns if col not in ['user_id', 'seat_id', 'overall_rating']])}")

    # ステップ3: データ分割
    print("\n" + "=" * 50)
    print("ステップ3: データ分割")
    print("=" * 50)

    X, y, feature_names = preprocessor.prepare_training_data(features_df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y,
        train_ratio=config['training']['train_ratio'],
        val_ratio=config['training']['val_ratio'],
        random_seed=config['training']['random_seed']
    )

    print(f"✓ 訓練データ: {X_train.shape}")
    print(f"✓ 検証データ: {X_val.shape}")
    print(f"✓ テストデータ: {X_test.shape}")

    # ステップ4: モデル訓練
    print("\n" + "=" * 50)
    print("ステップ4: モデル訓練")
    print("=" * 50)

    model_type = config['model']['type']
    model_params = config['model']['params']

    print(f"モデルタイプ: {model_type}")
    print(f"パラメータ: {model_params}")

    recommender = create_recommender(model_type, **model_params)
    metrics = recommender.train(X_train, y_train, X_val, y_val, feature_names)

    print("\n訓練結果:")
    print(f"  訓練 MSE: {metrics['train_mse']:.4f}")
    print(f"  訓練 MAE: {metrics['train_mae']:.4f}")
    print(f"  訓練 R²: {metrics['train_r2']:.4f}")

    if 'val_mse' in metrics:
        print(f"\n  検証 MSE: {metrics['val_mse']:.4f}")
        print(f"  検証 MAE: {metrics['val_mae']:.4f}")
        print(f"  検証 R²: {metrics['val_r2']:.4f}")

    # ステップ5: テストデータで評価
    print("\n" + "=" * 50)
    print("ステップ5: テストデータで評価")
    print("=" * 50)

    test_pred = recommender.predict(X_test)
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    test_mse = mean_squared_error(y_test, test_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_test, test_pred)

    print(f"テスト MSE: {test_mse:.4f}")
    print(f"テスト MAE: {test_mae:.4f}")
    print(f"テスト RMSE: {test_rmse:.4f}")
    print(f"テスト R²: {test_r2:.4f}")

    # 特徴量重要度を表示
    importance_df = recommender.get_feature_importance()
    if importance_df is not None:
        print("\n" + "=" * 50)
        print("特徴量重要度 Top 10")
        print("=" * 50)
        print(importance_df.head(10).to_string(index=False))

    # ステップ6: モデル保存
    print("\n" + "=" * 50)
    print("ステップ6: モデル保存")
    print("=" * 50)

    model_path = f"{config['output']['model_dir']}/seat_recommender.pkl"
    preprocessor_path = f"{config['output']['model_dir']}/preprocessor.pkl"

    recommender.save(model_path)
    preprocessor.save(preprocessor_path)

    print(f"✓ モデル保存: {model_path}")
    print(f"✓ 前処理器保存: {preprocessor_path}")

    # 評価結果を保存
    results = {
        'config': config,
        'metrics': {
            'train_mse': float(metrics['train_mse']),
            'train_mae': float(metrics['train_mae']),
            'train_r2': float(metrics['train_r2']),
            'val_mse': float(metrics.get('val_mse', 0)),
            'val_mae': float(metrics.get('val_mae', 0)),
            'val_r2': float(metrics.get('val_r2', 0)),
            'test_mse': float(test_mse),
            'test_mae': float(test_mae),
            'test_rmse': float(test_rmse),
            'test_r2': float(test_r2)
        },
        'feature_importance': importance_df.head(20).to_dict('records') if importance_df is not None else None
    }

    results_path = f"{config['output']['output_dir']}/training_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✓ 訓練結果保存: {results_path}")

    print("\n" + "=" * 50)
    print("訓練完了！")
    print("=" * 50)

    return recommender, preprocessor, features_df, results


def main():
    parser = argparse.ArgumentParser(description='座席推薦モデルの訓練')

    parser.add_argument('--config', type=str, help='設定ファイルのパス')
    parser.add_argument('--model-type', type=str, choices=['random_forest', 'gradient_boosting', 'neural_network', 'linear'],
                       help='モデルタイプ')
    parser.add_argument('--generate-new', action='store_true', help='新しいデータを生成')
    parser.add_argument('--num-seats', type=int, help='座席数')
    parser.add_argument('--num-users', type=int, help='ユーザー数')

    args = parser.parse_args()

    kwargs = {}
    if args.model_type:
        kwargs['model.type'] = args.model_type
    if args.generate_new:
        kwargs['data.generate_new'] = True
    if args.num_seats:
        kwargs['data.num_seats'] = args.num_seats
    if args.num_users:
        kwargs['data.num_users'] = args.num_users

    train_model(config_path=args.config, **kwargs)


if __name__ == "__main__":
    main()
