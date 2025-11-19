"""
座席配置最適化スクリプト

複数ユーザーに対して、全体の満足度を最大化する座席割り当てを実行します。
"""

import os
import sys
import argparse
import json
import pandas as pd

# srcディレクトリをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from model import BaseSeatRecommender
from preprocessor import DataPreprocessor
from optimizer import SeatAssignmentOptimizer, visualize_assignment


def print_header(text):
    """ヘッダーを表示"""
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='座席配置最適化')

    parser.add_argument('--model', type=str,
                       default='seat_recommendation/models/seat_recommender.pkl',
                       help='モデルファイルのパス')
    parser.add_argument('--preprocessor', type=str,
                       default='seat_recommendation/models/preprocessor.pkl',
                       help='前処理器ファイルのパス')
    parser.add_argument('--data-dir', type=str,
                       default='seat_recommendation/data',
                       help='データディレクトリ')
    parser.add_argument('--num-users', type=int, default=30,
                       help='割り当て対象のユーザー数')
    parser.add_argument('--method', type=str, default='hungarian',
                       choices=['hungarian', 'greedy', 'compare'],
                       help='最適化手法（hungarian: 全体最適, greedy: 高速, compare: 比較）')
    parser.add_argument('--occupied-seats', type=int, nargs='+',
                       help='既に使用中の座席ID')
    parser.add_argument('--output', type=str,
                       help='結果を保存するJSONファイルのパス')

    args = parser.parse_args()

    print_header("座席配置最適化システム")

    # モデルと前処理器を読み込み
    print("モデルを読み込んでいます...")
    try:
        recommender = BaseSeatRecommender.load(args.model)
        preprocessor = DataPreprocessor.load(args.preprocessor)
        print("✓ モデル読み込み完了")
    except FileNotFoundError as e:
        print(f"\nエラー: モデルファイルが見つかりません")
        print(f"まず以下を実行してモデルを訓練してください:")
        print(f"  python src/train.py --config config/config.yaml")
        sys.exit(1)

    # データを読み込み
    print("データを読み込んでいます...")
    seat_info = pd.read_csv(f"{args.data_dir}/seat_info.csv")
    env_data = pd.read_csv(f"{args.data_dir}/environment_data.csv")
    user_profiles = pd.read_csv(f"{args.data_dir}/user_profiles.csv")
    print("✓ データ読み込み完了")

    # 特徴量作成
    print("特徴量を作成しています...")
    features_df = preprocessor.create_features(
        user_profiles=user_profiles,
        seat_info=seat_info,
        env_data=env_data,
        user_ratings=None
    )
    print("✓ 特徴量作成完了")

    # 最適化器を作成
    optimizer = SeatAssignmentOptimizer(recommender)

    # 割り当て対象のユーザーを選択
    available_users = user_profiles['user_id'].tolist()
    num_users = min(args.num_users, len(available_users))
    target_users = available_users[:num_users]

    # 使用中の座席
    occupied_seats = set(args.occupied_seats) if args.occupied_seats else set()

    print(f"\n割り当て設定:")
    print(f"  対象ユーザー数: {num_users}")
    print(f"  総座席数: {len(seat_info)}")
    print(f"  使用中の座席数: {len(occupied_seats)}")
    print(f"  利用可能座席数: {len(seat_info) - len(occupied_seats)}")
    print(f"  最適化手法: {args.method}")

    # 最適化を実行
    print_header("最適化実行中...")

    if args.method == 'compare':
        # 複数手法を比較
        results = optimizer.compare_methods(
            user_ids=target_users,
            features_df=features_df,
            occupied_seats=occupied_seats
        )

        print_header("最適化結果の比較")

        for method, result in results.items():
            print(f"\n【{method.upper()}法】")
            print(f"  合計スコア: {result['total_score']:.2f}")
            print(f"  平均スコア: {result['average_score']:.2f}")
            print(f"  最小スコア: {result['min_score']:.2f}")
            print(f"  最大スコア: {result['max_score']:.2f}")

        # 最良の手法を選択
        best_method = max(results.items(), key=lambda x: x[1]['total_score'])[0]
        print(f"\n最良の手法: {best_method.upper()}法")
        assignments = results[best_method]['assignments']

    elif args.method == 'hungarian':
        # ハンガリアン法
        assignments = optimizer.optimize_hungarian(
            user_ids=target_users,
            features_df=features_df,
            occupied_seats=occupied_seats
        )

        total_score = sum(a['predicted_score'] for a in assignments.values())
        avg_score = total_score / len(assignments)

        print(f"\n✓ 最適化完了")
        print(f"  合計スコア: {total_score:.2f}")
        print(f"  平均スコア: {avg_score:.2f}")

    else:  # greedy
        # 貪欲法
        assignments = optimizer.optimize_greedy(
            user_ids=target_users,
            features_df=features_df,
            occupied_seats=occupied_seats
        )

        total_score = sum(a['predicted_score'] for a in assignments.values())
        avg_score = total_score / len(assignments)

        print(f"\n✓ 最適化完了")
        print(f"  合計スコア: {total_score:.2f}")
        print(f"  平均スコア: {avg_score:.2f}")

    # 座席配置を視覚化
    num_rows = seat_info['row'].max() + 1
    num_cols = seat_info['col'].max() + 1

    visualization = visualize_assignment(
        assignments=assignments,
        num_rows=num_rows,
        num_cols=num_cols,
        user_profiles=user_profiles
    )
    print(visualization)

    # 統計情報を表示
    print_header("満足度統計")

    scores = [a['predicted_score'] for a in assignments.values()]
    print(f"合計満足度: {sum(scores):.2f}")
    print(f"平均満足度: {sum(scores) / len(scores):.2f}")
    print(f"最小満足度: {min(scores):.2f}")
    print(f"最大満足度: {max(scores):.2f}")
    print(f"標準偏差: {pd.Series(scores).std():.2f}")

    # 適合度の統計
    temp_compat = [a['temp_compatibility'] for a in assignments.values()]
    noise_compat = [a['noise_compatibility'] for a in assignments.values()]

    print(f"\n温度適合度: 平均 {sum(temp_compat) / len(temp_compat):.2%}")
    print(f"騒音適合度: 平均 {sum(noise_compat) / len(noise_compat):.2%}")

    # 結果を保存
    if args.output:
        output_data = {
            'method': args.method,
            'num_users': num_users,
            'occupied_seats': list(occupied_seats),
            'total_score': sum(scores),
            'average_score': sum(scores) / len(scores),
            'assignments': {
                str(user_id): {
                    'seat_id': int(assignment['seat_id']),
                    'predicted_score': float(assignment['predicted_score']),
                    'seat_row': int(assignment['seat_row']),
                    'seat_col': int(assignment['seat_col']),
                    'temp_compatibility': float(assignment['temp_compatibility']),
                    'noise_compatibility': float(assignment['noise_compatibility']),
                    'light_compatibility': float(assignment['light_compatibility']),
                    'humidity_compatibility': float(assignment['humidity_compatibility']),
                }
                for user_id, assignment in assignments.items()
            }
        }

        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"\n✓ 結果を保存: {args.output}")

    print_header("最適化完了")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n中断されました。")
    except Exception as e:
        print(f"\n\nエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
