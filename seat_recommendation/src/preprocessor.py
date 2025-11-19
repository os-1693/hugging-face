"""
データ前処理モジュール

生データを機械学習モデルに適した形式に変換します。
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, Dict, Optional
import joblib


class DataPreprocessor:
    """データ前処理クラス"""

    def __init__(self):
        self.env_scaler = StandardScaler()
        self.user_scaler = StandardScaler()
        self.fitted = False

    def create_features(
        self,
        user_profiles: pd.DataFrame,
        seat_info: pd.DataFrame,
        env_data: pd.DataFrame,
        user_ratings: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        特徴量を作成

        Args:
            user_profiles: ユーザープロファイル
            seat_info: 座席情報
            env_data: 環境データ
            user_ratings: 評価データ（任意）

        Returns:
            特徴量DataFrame
        """
        # 環境データの集約（各座席の平均値を計算）
        env_agg = env_data.groupby('seat_id').agg({
            'temperature': ['mean', 'std'],
            'humidity': ['mean', 'std'],
            'illuminance': ['mean', 'std'],
            'noise_level': ['mean', 'std'],
            'co2_level': ['mean', 'std']
        }).reset_index()

        # カラム名をフラット化
        env_agg.columns = ['seat_id'] + [
            f'{col[0]}_{col[1]}' for col in env_agg.columns[1:]
        ]

        # 座席情報と環境データをマージ
        seat_features = seat_info.merge(env_agg, on='seat_id')

        # ユーザーと座席の全組み合わせを作成
        user_seat_combinations = []

        for _, user in user_profiles.iterrows():
            for _, seat in seat_features.iterrows():
                # ユーザーの好みと座席の特性の適合度を計算（ゼロ除算対策）
                temp_diff = abs(user['preferred_temperature'] - seat['temperature_mean'])
                temp_score = max(0, 1 - temp_diff / max(user['temperature_tolerance'], 1e-6))

                humidity_diff = abs(user['preferred_humidity'] - seat['humidity_mean'])
                humidity_score = max(0, 1 - humidity_diff / max(user['humidity_tolerance'], 1e-6))

                light_diff = abs(user['preferred_illuminance'] - seat['illuminance_mean'])
                light_score = max(0, 1 - light_diff / max(user['illuminance_tolerance'], 1e-6))

                noise_diff = max(0, seat['noise_level_mean'] - user['max_acceptable_noise'])
                noise_score = max(0, 1 - noise_diff / max(user['noise_tolerance'], 1e-6))

                # 特徴ベクトルを作成
                feature_dict = {
                    'user_id': user['user_id'],
                    'seat_id': seat['seat_id'],

                    # ユーザー特徴
                    'user_preferred_temp': user['preferred_temperature'],
                    'user_temp_tolerance': user['temperature_tolerance'],
                    'user_preferred_humidity': user['preferred_humidity'],
                    'user_humidity_tolerance': user['humidity_tolerance'],
                    'user_preferred_light': user['preferred_illuminance'],
                    'user_light_tolerance': user['illuminance_tolerance'],
                    'user_max_noise': user['max_acceptable_noise'],
                    'user_noise_tolerance': user['noise_tolerance'],
                    'user_prefers_window': user['prefers_window'],
                    'user_prefers_quiet': user['prefers_quiet'],
                    'user_needs_monitor': user['needs_monitor'],
                    'user_prefers_standing': user['prefers_standing_desk'],

                    # 座席特徴
                    'seat_row': seat['row'],
                    'seat_col': seat['col'],
                    'seat_is_window': seat['is_window'],
                    'seat_near_ac': seat['near_ac'],
                    'seat_near_entrance': seat['near_entrance'],
                    'seat_quiet_area': seat['quiet_area'],
                    'seat_has_monitor': seat['has_monitor'],
                    'seat_has_standing': seat['has_standing_desk'],

                    # 環境特徴
                    'seat_temp_mean': seat['temperature_mean'],
                    'seat_temp_std': seat['temperature_std'],
                    'seat_humidity_mean': seat['humidity_mean'],
                    'seat_humidity_std': seat['humidity_std'],
                    'seat_light_mean': seat['illuminance_mean'],
                    'seat_light_std': seat['illuminance_std'],
                    'seat_noise_mean': seat['noise_level_mean'],
                    'seat_noise_std': seat['noise_level_std'],
                    'seat_co2_mean': seat['co2_level_mean'],
                    'seat_co2_std': seat['co2_level_std'],

                    # 適合度スコア
                    'temp_compatibility': temp_score,
                    'humidity_compatibility': humidity_score,
                    'light_compatibility': light_score,
                    'noise_compatibility': noise_score,

                    # 好みの一致度
                    'window_match': int(user['prefers_window'] == seat['is_window']),
                    'quiet_match': int(user['prefers_quiet'] == seat['quiet_area']),
                    'monitor_match': int(user['needs_monitor'] == seat['has_monitor']),
                    'standing_match': int(user['prefers_standing_desk'] == seat['has_standing_desk'])
                }

                user_seat_combinations.append(feature_dict)

        features_df = pd.DataFrame(user_seat_combinations)

        # 評価データがある場合はマージ
        if user_ratings is not None:
            # 評価の平均を計算
            ratings_agg = user_ratings.groupby(['user_id', 'seat_id']).agg({
                'overall_rating': 'mean',
                'temperature_rating': 'mean',
                'noise_rating': 'mean',
                'light_rating': 'mean'
            }).reset_index()

            features_df = features_df.merge(
                ratings_agg,
                on=['user_id', 'seat_id'],
                how='left'
            )

        return features_df

    def prepare_training_data(
        self,
        features_df: pd.DataFrame,
        target_col: str = 'overall_rating'
    ) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        訓練データを準備

        Args:
            features_df: 特徴量DataFrame
            target_col: ターゲット列名

        Returns:
            (X, y, feature_names)のタプル
        """
        # ターゲットがあるレコードのみ使用
        train_df = features_df.dropna(subset=[target_col])

        # 特徴量カラムを選択
        feature_cols = [col for col in train_df.columns
                       if col not in ['user_id', 'seat_id', 'overall_rating',
                                     'temperature_rating', 'noise_rating', 'light_rating']]

        X = train_df[feature_cols].values
        y = train_df[target_col].values

        # スケーリング
        if not self.fitted:
            X = self.env_scaler.fit_transform(X)
            self.fitted = True
        else:
            X = self.env_scaler.transform(X)

        return X, y, feature_cols

    def prepare_inference_data(
        self,
        features_df: pd.DataFrame
    ) -> Tuple[np.ndarray, list]:
        """
        推論データを準備

        Args:
            features_df: 特徴量DataFrame

        Returns:
            (X, feature_names)のタプル
        """
        # 特徴量カラムを選択
        feature_cols = [col for col in features_df.columns
                       if col not in ['user_id', 'seat_id', 'overall_rating',
                                     'temperature_rating', 'noise_rating', 'light_rating']]

        X = features_df[feature_cols].values

        # スケーリング
        if self.fitted:
            X = self.env_scaler.transform(X)
        else:
            raise ValueError("Preprocessor not fitted. Call prepare_training_data first.")

        return X, feature_cols

    def save(self, filepath: str):
        """前処理器を保存（joblibを使用）"""
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> 'DataPreprocessor':
        """前処理器を読み込み（joblibを使用）"""
        return joblib.load(filepath)


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    データを訓練、検証、テストに分割

    Args:
        X: 特徴量
        y: ターゲット
        train_ratio: 訓練データの割合
        val_ratio: 検証データの割合
        random_seed: ランダムシード

    Returns:
        (X_train, X_val, X_test, y_train, y_val, y_test)のタプル
    """
    from sklearn.model_selection import train_test_split

    # まず訓練とテストに分割
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=1 - train_ratio - val_ratio, random_state=random_seed
    )

    # 訓練と検証に分割
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio_adjusted, random_state=random_seed
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
