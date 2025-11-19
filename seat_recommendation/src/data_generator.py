"""
サンプルデータ生成モジュール

室内環境データ、座席情報、ユーザーアンケートデータを生成します。
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict
import json


class EnvironmentDataGenerator:
    """室内環境データ生成クラス"""

    def __init__(self, num_seats: int = 50, num_timestamps: int = 1000, random_seed: int = 42):
        """
        Args:
            num_seats: 座席数
            num_timestamps: データポイント数（タイムスタンプ）
            random_seed: ランダムシード
        """
        self.num_seats = num_seats
        self.num_timestamps = num_timestamps
        np.random.seed(random_seed)

    def generate_seat_info(self) -> pd.DataFrame:
        """
        座席の固定情報を生成

        Returns:
            座席情報のDataFrame
        """
        seats_data = []

        for seat_id in range(self.num_seats):
            # 5x10のグリッドレイアウトを想定
            row = seat_id // 10
            col = seat_id % 10

            # 窓際かどうか（列0と9）
            is_window = (col == 0 or col == 9)

            # 空調の近く（行0と4）
            near_ac = (row == 0 or row == 4)

            # 出入り口の近く（row 4の右端）
            near_entrance = (row == 4 and col >= 8)

            # 静かなエリア（後方）
            quiet_area = (row <= 1)

            seats_data.append({
                'seat_id': seat_id,
                'row': row,
                'col': col,
                'is_window': int(is_window),
                'near_ac': int(near_ac),
                'near_entrance': int(near_entrance),
                'quiet_area': int(quiet_area),
                'has_monitor': int(np.random.random() > 0.3),  # 70%の座席にモニター
                'has_standing_desk': int(np.random.random() > 0.7)  # 30%がスタンディングデスク
            })

        return pd.DataFrame(seats_data)

    def generate_environment_data(self, seat_info: pd.DataFrame) -> pd.DataFrame:
        """
        時系列の室内環境データを生成

        Args:
            seat_info: 座席情報DataFrame

        Returns:
            環境データのDataFrame
        """
        env_data = []

        for timestamp in range(self.num_timestamps):
            # 時刻による変動（0-23時）
            hour = (timestamp % 24)

            for _, seat in seat_info.iterrows():
                seat_id = seat['seat_id']

                # 基本温度: 22-26度
                base_temp = 23.5 + np.random.normal(0, 0.5)

                # 窓際は温度変動が大きい
                if seat['is_window']:
                    temp_variation = np.sin(hour * np.pi / 12) * 2  # 日照による変動
                else:
                    temp_variation = 0

                # 空調の近くは少し涼しい
                if seat['near_ac']:
                    temp_variation -= 1

                temperature = base_temp + temp_variation + np.random.normal(0, 0.3)

                # 湿度: 40-60%
                humidity = 50 + np.random.normal(0, 5) + (np.sin(hour * np.pi / 12) * 5)

                # 照度: 300-800 lux（窓際は明るい）
                base_light = 500
                if seat['is_window']:
                    # 日中（8-18時）は窓際が明るい
                    if 8 <= hour <= 18:
                        base_light += 200 * np.sin((hour - 8) * np.pi / 10)

                illuminance = max(200, base_light + np.random.normal(0, 50))

                # 騒音レベル: 30-70 dB（出入り口付近は騒がしい）
                base_noise = 45
                if seat['near_entrance']:
                    base_noise += 10
                if seat['quiet_area']:
                    base_noise -= 5

                # 時間帯による変動（休憩時間は騒がしい）
                if hour in [12, 15, 18]:  # 休憩時間
                    base_noise += 5

                noise_level = max(30, base_noise + np.random.normal(0, 3))

                # CO2濃度: 400-1200 ppm（人が多いと高くなる）
                occupancy_factor = (np.sin(hour * np.pi / 12) + 1) / 2  # 0-1の範囲
                co2_level = 400 + 400 * occupancy_factor + np.random.normal(0, 50)

                env_data.append({
                    'timestamp': timestamp,
                    'seat_id': seat_id,
                    'temperature': round(temperature, 2),
                    'humidity': round(max(30, min(70, humidity)), 2),
                    'illuminance': round(illuminance, 2),
                    'noise_level': round(noise_level, 2),
                    'co2_level': round(co2_level, 2),
                    'hour': hour
                })

        return pd.DataFrame(env_data)


class UserDataGenerator:
    """ユーザーアンケートデータ生成クラス"""

    def __init__(self, num_users: int = 200, random_seed: int = 42):
        """
        Args:
            num_users: ユーザー数
            random_seed: ランダムシード
        """
        self.num_users = num_users
        np.random.seed(random_seed)

    def generate_user_profiles(self) -> pd.DataFrame:
        """
        ユーザープロファイルを生成

        Returns:
            ユーザープロファイルのDataFrame
        """
        users_data = []

        for user_id in range(self.num_users):
            # ユーザータイプを定義（クラスタリング）
            user_type = np.random.choice(['cold_sensitive', 'heat_sensitive',
                                         'noise_sensitive', 'light_sensitive',
                                         'balanced'], p=[0.2, 0.2, 0.2, 0.2, 0.2])

            # 好みの温度（20-26度）
            if user_type == 'cold_sensitive':
                preferred_temp = np.random.uniform(24, 26)
                temp_tolerance = np.random.uniform(0.5, 1.0)
            elif user_type == 'heat_sensitive':
                preferred_temp = np.random.uniform(20, 22)
                temp_tolerance = np.random.uniform(0.5, 1.0)
            else:
                preferred_temp = np.random.uniform(22, 24)
                temp_tolerance = np.random.uniform(1.0, 2.0)

            # 好みの湿度（40-60%）
            preferred_humidity = np.random.uniform(45, 55)
            humidity_tolerance = np.random.uniform(5, 10)

            # 好みの照度（300-700 lux）
            if user_type == 'light_sensitive':
                preferred_light = np.random.uniform(300, 450)
                light_tolerance = np.random.uniform(50, 100)
            else:
                preferred_light = np.random.uniform(500, 700)
                light_tolerance = np.random.uniform(100, 200)

            # 騒音感度（30-60 dB以下を好む）
            if user_type == 'noise_sensitive':
                max_acceptable_noise = np.random.uniform(35, 45)
                noise_tolerance = np.random.uniform(2, 5)
            else:
                max_acceptable_noise = np.random.uniform(50, 60)
                noise_tolerance = np.random.uniform(5, 10)

            # その他の好み
            prefers_window = int(np.random.random() > 0.5)
            prefers_quiet = int(np.random.random() > 0.6)
            needs_monitor = int(np.random.random() > 0.4)
            prefers_standing = int(np.random.random() > 0.8)

            users_data.append({
                'user_id': user_id,
                'user_type': user_type,
                'preferred_temperature': round(preferred_temp, 2),
                'temperature_tolerance': round(temp_tolerance, 2),
                'preferred_humidity': round(preferred_humidity, 2),
                'humidity_tolerance': round(humidity_tolerance, 2),
                'preferred_illuminance': round(preferred_light, 2),
                'illuminance_tolerance': round(light_tolerance, 2),
                'max_acceptable_noise': round(max_acceptable_noise, 2),
                'noise_tolerance': round(noise_tolerance, 2),
                'prefers_window': prefers_window,
                'prefers_quiet': prefers_quiet,
                'needs_monitor': needs_monitor,
                'prefers_standing_desk': prefers_standing
            })

        return pd.DataFrame(users_data)

    def generate_user_ratings(
        self,
        num_ratings: int,
        num_seats: int,
        user_profiles: pd.DataFrame,
        seat_info: pd.DataFrame,
        env_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        ユーザーの座席評価履歴を生成（現実的な評価を生成）

        Args:
            num_ratings: 評価数
            num_seats: 座席数
            user_profiles: ユーザープロファイル
            seat_info: 座席情報
            env_data: 環境データ

        Returns:
            評価履歴のDataFrame
        """
        ratings_data = []

        # 環境データの集約（座席ごとの平均）
        env_agg = env_data.groupby('seat_id').agg({
            'temperature': 'mean',
            'humidity': 'mean',
            'illuminance': 'mean',
            'noise_level': 'mean',
            'co2_level': 'mean'
        }).reset_index()

        # 座席情報とマージ
        seat_features = seat_info.merge(env_agg, on='seat_id')

        for _ in range(num_ratings):
            user_id = np.random.randint(0, self.num_users)
            seat_id = np.random.randint(0, num_seats)

            user = user_profiles[user_profiles['user_id'] == user_id].iloc[0]
            seat = seat_features[seat_features['seat_id'] == seat_id].iloc[0]

            # 現実的な評価を計算（ユーザーの好みと座席特性に基づく）
            # 温度の適合度（0-1）
            temp_diff = abs(user['preferred_temperature'] - seat['temperature'])
            temp_score = max(0, 1 - temp_diff / max(user['temperature_tolerance'] * 3, 1e-6))

            # 湿度の適合度
            humidity_diff = abs(user['preferred_humidity'] - seat['humidity'])
            humidity_score = max(0, 1 - humidity_diff / max(user['humidity_tolerance'] * 2, 1e-6))

            # 照度の適合度
            light_diff = abs(user['preferred_illuminance'] - seat['illuminance'])
            light_score = max(0, 1 - light_diff / max(user['illuminance_tolerance'] * 2, 1e-6))

            # 騒音の適合度
            noise_diff = max(0, seat['noise_level'] - user['max_acceptable_noise'])
            noise_score = max(0, 1 - noise_diff / max(user['noise_tolerance'] * 3, 1e-6))

            # 好みの一致度（窓際、静かなエリアなど）
            pref_match = 0
            if user['prefers_window'] == seat['is_window']:
                pref_match += 0.25
            if user['prefers_quiet'] == seat['quiet_area']:
                pref_match += 0.25
            if user['needs_monitor'] == seat['has_monitor']:
                pref_match += 0.25
            if user['prefers_standing_desk'] == seat['has_standing_desk']:
                pref_match += 0.25

            # 総合評価（1-5スケール）
            # 各要素を加重平均
            base_score = (
                temp_score * 0.3 +
                humidity_score * 0.15 +
                light_score * 0.15 +
                noise_score * 0.25 +
                pref_match * 0.15
            )

            # 1-5の範囲に変換し、ノイズを追加
            overall_rating = np.clip(base_score * 5 + np.random.normal(0, 0.3), 1, 5)
            temperature_rating = np.clip(temp_score * 5 + np.random.normal(0, 0.4), 1, 5)
            noise_rating = np.clip(noise_score * 5 + np.random.normal(0, 0.4), 1, 5)
            light_rating = np.clip(light_score * 5 + np.random.normal(0, 0.4), 1, 5)

            ratings_data.append({
                'user_id': user_id,
                'seat_id': seat_id,
                'overall_rating': round(overall_rating, 2),
                'temperature_rating': round(temperature_rating, 2),
                'noise_rating': round(noise_rating, 2),
                'light_rating': round(light_rating, 2),
                'timestamp': np.random.randint(0, 1000)
            })

        return pd.DataFrame(ratings_data)


def generate_all_data(
    num_seats: int = 50,
    num_users: int = 200,
    num_timestamps: int = 1000,
    num_ratings: int = 5000,
    output_dir: str = "data"
) -> Dict[str, pd.DataFrame]:
    """
    全てのデータを生成して保存

    Args:
        num_seats: 座席数
        num_users: ユーザー数
        num_timestamps: 環境データのタイムスタンプ数
        num_ratings: 評価データ数
        output_dir: 出力ディレクトリ

    Returns:
        生成されたデータのディクショナリ
    """
    print("データ生成を開始します...")

    # 環境データ生成
    env_gen = EnvironmentDataGenerator(num_seats, num_timestamps)
    seat_info = env_gen.generate_seat_info()
    env_data = env_gen.generate_environment_data(seat_info)

    print(f"✓ 座席情報: {len(seat_info)} 座席")
    print(f"✓ 環境データ: {len(env_data)} レコード")

    # ユーザーデータ生成
    user_gen = UserDataGenerator(num_users)
    user_profiles = user_gen.generate_user_profiles()
    user_ratings = user_gen.generate_user_ratings(
        num_ratings, num_seats, user_profiles, seat_info, env_data
    )

    print(f"✓ ユーザープロファイル: {len(user_profiles)} ユーザー")
    print(f"✓ 評価データ: {len(user_ratings)} 評価（現実的な適合度に基づく）")

    # データ保存
    import os
    os.makedirs(output_dir, exist_ok=True)

    seat_info.to_csv(f"{output_dir}/seat_info.csv", index=False)
    env_data.to_csv(f"{output_dir}/environment_data.csv", index=False)
    user_profiles.to_csv(f"{output_dir}/user_profiles.csv", index=False)
    user_ratings.to_csv(f"{output_dir}/user_ratings.csv", index=False)

    print(f"\n✓ 全データを {output_dir}/ に保存しました")

    # 統計情報を表示
    print("\n=== データ統計 ===")
    print(f"座席数: {num_seats}")
    print(f"ユーザー数: {num_users}")
    print(f"環境データポイント数: {len(env_data)}")
    print(f"ユーザー評価数: {len(user_ratings)}")

    return {
        'seat_info': seat_info,
        'environment_data': env_data,
        'user_profiles': user_profiles,
        'user_ratings': user_ratings
    }


if __name__ == "__main__":
    # サンプルデータ生成
    data = generate_all_data(
        num_seats=50,
        num_users=200,
        num_timestamps=1000,
        num_ratings=5000,
        output_dir="data"
    )

    print("\n=== サンプルデータプレビュー ===")
    print("\n[座席情報]")
    print(data['seat_info'].head())
    print("\n[環境データ]")
    print(data['environment_data'].head())
    print("\n[ユーザープロファイル]")
    print(data['user_profiles'].head())
    print("\n[評価データ]")
    print(data['user_ratings'].head())
