"""
座席配置最適化モジュール

複数ユーザーに対して、全体の満足度を最大化する座席割り当てを行います。
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Set
from scipy.optimize import linear_sum_assignment
from model import BaseSeatRecommender


class SeatAssignmentOptimizer:
    """
    座席配置最適化クラス

    複数のユーザーに対して最適な座席割り当てを計算します。
    """

    def __init__(self, recommender: BaseSeatRecommender):
        """
        Args:
            recommender: 訓練済みの座席推薦モデル
        """
        self.recommender = recommender

    def optimize_hungarian(
        self,
        user_ids: List[int],
        features_df: pd.DataFrame,
        occupied_seats: Optional[Set[int]] = None,
        reserved_assignments: Optional[Dict[int, int]] = None
    ) -> Dict[int, Dict]:
        """
        ハンガリアンアルゴリズムを使った最適割り当て

        全体の満足度（予測スコアの合計）を最大化する座席割り当てを計算します。

        Args:
            user_ids: 割り当て対象のユーザーIDリスト
            features_df: 特徴量DataFrame（全ユーザー・座席の組み合わせ）
            occupied_seats: 既に使用中の座席IDのセット
            reserved_assignments: 事前に決定済みの割り当て {user_id: seat_id}

        Returns:
            割り当て結果のディクショナリ {user_id: {seat_id, score, ...}}
        """
        if occupied_seats is None:
            occupied_seats = set()
        if reserved_assignments is None:
            reserved_assignments = {}

        # 利用可能な座席を取得
        all_seats = set(features_df['seat_id'].unique())
        available_seats = list(all_seats - occupied_seats - set(reserved_assignments.values()))

        # 割り当て対象のユーザー（予約済みを除く）
        users_to_assign = [uid for uid in user_ids if uid not in reserved_assignments]

        if len(users_to_assign) > len(available_seats):
            raise ValueError(
                f"座席数が不足しています。ユーザー数: {len(users_to_assign)}, "
                f"利用可能座席数: {len(available_seats)}"
            )

        # コスト行列を作成（最大化問題なので負の値を使用）
        cost_matrix = self._create_cost_matrix(
            users_to_assign,
            available_seats,
            features_df
        )

        # ハンガリアンアルゴリズムで最適割り当てを計算
        user_indices, seat_indices = linear_sum_assignment(cost_matrix)

        # 結果を整理
        assignments = {}

        # 予約済み割り当てを追加
        for user_id, seat_id in reserved_assignments.items():
            user_data = features_df[
                (features_df['user_id'] == user_id) &
                (features_df['seat_id'] == seat_id)
            ].iloc[0]

            # 予測スコアを計算
            feature_cols = [col for col in features_df.columns
                          if col not in ['user_id', 'seat_id', 'overall_rating',
                                       'temperature_rating', 'noise_rating', 'light_rating']]
            X = user_data[feature_cols].values.reshape(1, -1)
            predicted_score = self.recommender.predict(X)[0]

            assignments[user_id] = {
                'seat_id': seat_id,
                'predicted_score': predicted_score,
                'seat_row': user_data['seat_row'],
                'seat_col': user_data['seat_col'],
                'temp_compatibility': user_data['temp_compatibility'],
                'noise_compatibility': user_data['noise_compatibility'],
                'light_compatibility': user_data['light_compatibility'],
                'humidity_compatibility': user_data['humidity_compatibility'],
                'reserved': True
            }

        # 最適化された割り当てを追加
        for user_idx, seat_idx in zip(user_indices, seat_indices):
            user_id = users_to_assign[user_idx]
            seat_id = available_seats[seat_idx]

            user_data = features_df[
                (features_df['user_id'] == user_id) &
                (features_df['seat_id'] == seat_id)
            ].iloc[0]

            # 予測スコアを取得（コスト行列の負の値）
            predicted_score = -cost_matrix[user_idx, seat_idx]

            assignments[user_id] = {
                'seat_id': seat_id,
                'predicted_score': predicted_score,
                'seat_row': user_data['seat_row'],
                'seat_col': user_data['seat_col'],
                'temp_compatibility': user_data['temp_compatibility'],
                'noise_compatibility': user_data['noise_compatibility'],
                'light_compatibility': user_data['light_compatibility'],
                'humidity_compatibility': user_data['humidity_compatibility'],
                'reserved': False
            }

        return assignments

    def optimize_greedy(
        self,
        user_ids: List[int],
        features_df: pd.DataFrame,
        occupied_seats: Optional[Set[int]] = None,
        reserved_assignments: Optional[Dict[int, int]] = None,
        priority_users: Optional[List[int]] = None
    ) -> Dict[int, Dict]:
        """
        貪欲法による座席割り当て

        各ユーザーに対して順番に最良の座席を割り当てます。
        計算が高速ですが、全体最適ではありません。

        Args:
            user_ids: 割り当て対象のユーザーIDリスト
            features_df: 特徴量DataFrame
            occupied_seats: 既に使用中の座席IDのセット
            reserved_assignments: 事前に決定済みの割り当て
            priority_users: 優先的に割り当てるユーザーIDのリスト

        Returns:
            割り当て結果のディクショナリ
        """
        if occupied_seats is None:
            occupied_seats = set()
        if reserved_assignments is None:
            reserved_assignments = {}
        if priority_users is None:
            priority_users = []

        # 利用可能な座席
        all_seats = set(features_df['seat_id'].unique())
        available_seats = all_seats - occupied_seats - set(reserved_assignments.values())

        # ユーザーの順序を決定（優先ユーザーを先に）
        priority_set = set(priority_users)
        ordered_users = (
            [uid for uid in user_ids if uid in priority_set and uid not in reserved_assignments] +
            [uid for uid in user_ids if uid not in priority_set and uid not in reserved_assignments]
        )

        assignments = {}

        # 予約済み割り当てを追加
        for user_id, seat_id in reserved_assignments.items():
            user_data = features_df[
                (features_df['user_id'] == user_id) &
                (features_df['seat_id'] == seat_id)
            ].iloc[0]

            feature_cols = [col for col in features_df.columns
                          if col not in ['user_id', 'seat_id', 'overall_rating',
                                       'temperature_rating', 'noise_rating', 'light_rating']]
            X = user_data[feature_cols].values.reshape(1, -1)
            predicted_score = self.recommender.predict(X)[0]

            assignments[user_id] = {
                'seat_id': seat_id,
                'predicted_score': predicted_score,
                'seat_row': user_data['seat_row'],
                'seat_col': user_data['seat_col'],
                'temp_compatibility': user_data['temp_compatibility'],
                'noise_compatibility': user_data['noise_compatibility'],
                'light_compatibility': user_data['light_compatibility'],
                'humidity_compatibility': user_data['humidity_compatibility'],
                'reserved': True
            }

        # 各ユーザーに対して貪欲に座席を割り当て
        for user_id in ordered_users:
            if not available_seats:
                raise ValueError(f"座席が不足しています。ユーザーID {user_id} に割り当てできません。")

            # ユーザーのデータを取得
            user_data = features_df[
                (features_df['user_id'] == user_id) &
                (features_df['seat_id'].isin(available_seats))
            ]

            # 予測スコアを計算
            feature_cols = [col for col in features_df.columns
                          if col not in ['user_id', 'seat_id', 'overall_rating',
                                       'temperature_rating', 'noise_rating', 'light_rating']]
            X = user_data[feature_cols].values
            scores = self.recommender.predict(X)

            # 最良の座席を選択
            best_idx = np.argmax(scores)
            best_seat_data = user_data.iloc[best_idx]
            best_seat_id = best_seat_data['seat_id']

            assignments[user_id] = {
                'seat_id': best_seat_id,
                'predicted_score': scores[best_idx],
                'seat_row': best_seat_data['seat_row'],
                'seat_col': best_seat_data['seat_col'],
                'temp_compatibility': best_seat_data['temp_compatibility'],
                'noise_compatibility': best_seat_data['noise_compatibility'],
                'light_compatibility': best_seat_data['light_compatibility'],
                'humidity_compatibility': best_seat_data['humidity_compatibility'],
                'reserved': False
            }

            # 座席を利用可能リストから削除
            available_seats.remove(best_seat_id)

        return assignments

    def _create_cost_matrix(
        self,
        user_ids: List[int],
        seat_ids: List[int],
        features_df: pd.DataFrame
    ) -> np.ndarray:
        """
        コスト行列を作成

        Args:
            user_ids: ユーザーIDリスト
            seat_ids: 座席IDリスト
            features_df: 特徴量DataFrame

        Returns:
            コスト行列（負の予測スコア）
        """
        cost_matrix = np.zeros((len(user_ids), len(seat_ids)))

        feature_cols = [col for col in features_df.columns
                       if col not in ['user_id', 'seat_id', 'overall_rating',
                                    'temperature_rating', 'noise_rating', 'light_rating']]

        for i, user_id in enumerate(user_ids):
            for j, seat_id in enumerate(seat_ids):
                # ユーザーと座席の組み合わせのデータを取得
                data = features_df[
                    (features_df['user_id'] == user_id) &
                    (features_df['seat_id'] == seat_id)
                ]

                if len(data) == 0:
                    # データがない場合は大きなコスト
                    cost_matrix[i, j] = 1000
                else:
                    # 予測スコアを計算（最大化したいので負の値）
                    X = data[feature_cols].values.reshape(1, -1)
                    predicted_score = self.recommender.predict(X)[0]
                    cost_matrix[i, j] = -predicted_score

        return cost_matrix

    def compare_methods(
        self,
        user_ids: List[int],
        features_df: pd.DataFrame,
        occupied_seats: Optional[Set[int]] = None,
        reserved_assignments: Optional[Dict[int, int]] = None
    ) -> Dict[str, Dict]:
        """
        複数の最適化手法を比較

        Args:
            user_ids: ユーザーIDリスト
            features_df: 特徴量DataFrame
            occupied_seats: 使用中の座席
            reserved_assignments: 予約済み割り当て

        Returns:
            各手法の結果と統計情報
        """
        results = {}

        # ハンガリアン法
        hungarian_assignments = self.optimize_hungarian(
            user_ids, features_df, occupied_seats, reserved_assignments
        )
        hungarian_total = sum(a['predicted_score'] for a in hungarian_assignments.values())
        hungarian_avg = hungarian_total / len(hungarian_assignments)

        results['hungarian'] = {
            'assignments': hungarian_assignments,
            'total_score': hungarian_total,
            'average_score': hungarian_avg,
            'min_score': min(a['predicted_score'] for a in hungarian_assignments.values()),
            'max_score': max(a['predicted_score'] for a in hungarian_assignments.values())
        }

        # 貪欲法
        greedy_assignments = self.optimize_greedy(
            user_ids, features_df, occupied_seats, reserved_assignments
        )
        greedy_total = sum(a['predicted_score'] for a in greedy_assignments.values())
        greedy_avg = greedy_total / len(greedy_assignments)

        results['greedy'] = {
            'assignments': greedy_assignments,
            'total_score': greedy_total,
            'average_score': greedy_avg,
            'min_score': min(a['predicted_score'] for a in greedy_assignments.values()),
            'max_score': max(a['predicted_score'] for a in greedy_assignments.values())
        }

        return results


def visualize_assignment(
    assignments: Dict[int, Dict],
    num_rows: int = 5,
    num_cols: int = 10,
    user_profiles: Optional[pd.DataFrame] = None
) -> str:
    """
    座席割り当てを視覚化

    Args:
        assignments: 割り当て結果
        num_rows: 座席レイアウトの行数
        num_cols: 座席レイアウトの列数
        user_profiles: ユーザープロファイル（オプション）

    Returns:
        視覚化されたテキスト
    """
    # 座席マップを作成
    seat_map = {}
    for user_id, assignment in assignments.items():
        seat_id = assignment['seat_id']
        seat_map[seat_id] = user_id

    # グリッドを描画
    lines = []
    lines.append("\n" + "=" * 60)
    lines.append("座席配置図")
    lines.append("=" * 60 + "\n")

    for row in range(num_rows):
        row_display = []
        for col in range(num_cols):
            seat_id = row * num_cols + col
            if seat_id in seat_map:
                user_id = seat_map[seat_id]
                # ユーザーIDを表示
                row_display.append(f"U{user_id:03d}")
            else:
                row_display.append(" --- ")

        lines.append("  ".join(row_display))

    lines.append("\n" + "=" * 60)
    lines.append("割り当て詳細")
    lines.append("=" * 60 + "\n")

    # 割り当て詳細
    for user_id in sorted(assignments.keys()):
        assignment = assignments[user_id]
        lines.append(
            f"ユーザー {user_id:3d} → 座席 {assignment['seat_id']:3d} "
            f"(行{assignment['seat_row']}, 列{assignment['seat_col']}) "
            f"| スコア: {assignment['predicted_score']:.2f} "
            f"{'[予約済み]' if assignment['reserved'] else ''}"
        )

    return "\n".join(lines)
