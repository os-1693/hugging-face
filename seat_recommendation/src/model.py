"""
座席推薦モデル

複数の機械学習アルゴリズムを使った座席推薦モデルを実装します。
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
import joblib
from abc import ABC, abstractmethod

# 機械学習ライブラリ
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class BaseSeatRecommender(ABC):
    """座席推薦モデルの基底クラス"""

    def __init__(self):
        self.model = None
        self.feature_names = None
        self.trained = False

    @abstractmethod
    def build_model(self):
        """モデルを構築"""
        pass

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        モデルを訓練

        Args:
            X_train: 訓練データの特徴量
            y_train: 訓練データのターゲット
            X_val: 検証データの特徴量
            y_val: 検証データのターゲット
            feature_names: 特徴量名のリスト

        Returns:
            評価メトリクスのディクショナリ
        """
        if self.model is None:
            self.build_model()

        self.feature_names = feature_names
        self.model.fit(X_train, y_train)
        self.trained = True

        # 評価
        metrics = {}
        train_pred = self.model.predict(X_train)
        metrics['train_mse'] = mean_squared_error(y_train, train_pred)
        metrics['train_mae'] = mean_absolute_error(y_train, train_pred)
        metrics['train_r2'] = r2_score(y_train, train_pred)

        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            metrics['val_mse'] = mean_squared_error(y_val, val_pred)
            metrics['val_mae'] = mean_absolute_error(y_val, val_pred)
            metrics['val_r2'] = r2_score(y_val, val_pred)

        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """予測を実行"""
        if not self.trained:
            raise ValueError("Model not trained yet. Call train() first.")
        return self.model.predict(X)

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """特徴量の重要度を取得"""
        if not hasattr(self.model, 'feature_importances_'):
            return None

        if self.feature_names is None:
            return None

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        return importance_df

    def save(self, filepath: str):
        """モデルを保存（joblibを使用）"""
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> 'BaseSeatRecommender':
        """モデルを読み込み（joblibを使用）"""
        return joblib.load(filepath)


class RandomForestRecommender(BaseSeatRecommender):
    """ランダムフォレスト推薦モデル"""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        random_state: int = 42
    ):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state

    def build_model(self):
        """ランダムフォレストモデルを構築"""
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            random_state=self.random_state,
            n_jobs=-1
        )


class GradientBoostingRecommender(BaseSeatRecommender):
    """勾配ブースティング推薦モデル"""

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        random_state: int = 42
    ):
        super().__init__()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state

    def build_model(self):
        """勾配ブースティングモデルを構築"""
        self.model = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=self.random_state
        )


class NeuralNetworkRecommender(BaseSeatRecommender):
    """ニューラルネットワーク推薦モデル"""

    def __init__(
        self,
        hidden_layer_sizes: Tuple[int, ...] = (100, 50),
        activation: str = 'relu',
        learning_rate_init: float = 0.001,
        max_iter: int = 200,
        random_state: int = 42
    ):
        super().__init__()
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.random_state = random_state

    def build_model(self):
        """ニューラルネットワークモデルを構築"""
        self.model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
            random_state=self.random_state,
            early_stopping=True,
            validation_fraction=0.1
        )


class LinearRecommender(BaseSeatRecommender):
    """線形回帰推薦モデル"""

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def build_model(self):
        """線形モデルを構築"""
        if self.alpha > 0:
            self.model = Ridge(alpha=self.alpha)
        else:
            self.model = LinearRegression()


class SeatRecommendationSystem:
    """
    座席推薦システム

    複数のユーザーに対して最適な座席を推薦します。
    """

    def __init__(self, model: BaseSeatRecommender):
        """
        Args:
            model: 推薦モデル
        """
        self.model = model

    def recommend_seats(
        self,
        user_id: int,
        features_df: pd.DataFrame,
        top_k: int = 5,
        exclude_seats: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        特定ユーザーに対して座席を推薦

        Args:
            user_id: ユーザーID
            features_df: 特徴量DataFrame（全ユーザー・座席の組み合わせ）
            top_k: 推薦する座席数
            exclude_seats: 除外する座席IDのリスト

        Returns:
            推薦座席のDataFrame（座席ID、予測スコア、特徴を含む）
        """
        # 該当ユーザーのデータを抽出
        user_data = features_df[features_df['user_id'] == user_id].copy()

        if len(user_data) == 0:
            raise ValueError(f"User {user_id} not found in features_df")

        # 除外座席をフィルタ
        if exclude_seats:
            user_data = user_data[~user_data['seat_id'].isin(exclude_seats)]

        # 特徴量を準備
        feature_cols = [col for col in user_data.columns
                       if col not in ['user_id', 'seat_id', 'overall_rating',
                                     'temperature_rating', 'noise_rating', 'light_rating']]

        X = user_data[feature_cols].values

        # 予測
        predictions = self.model.predict(X)
        user_data['predicted_rating'] = predictions

        # Top-K座席を選択
        recommendations = user_data.nlargest(top_k, 'predicted_rating')[[
            'seat_id', 'predicted_rating',
            'seat_row', 'seat_col',
            'seat_temp_mean', 'seat_noise_mean',
            'temp_compatibility', 'noise_compatibility',
            'light_compatibility', 'humidity_compatibility'
        ]]

        return recommendations.reset_index(drop=True)

    def recommend_for_multiple_users(
        self,
        user_ids: List[int],
        features_df: pd.DataFrame,
        top_k: int = 5
    ) -> Dict[int, pd.DataFrame]:
        """
        複数ユーザーに対して座席を推薦

        Args:
            user_ids: ユーザーIDのリスト
            features_df: 特徴量DataFrame
            top_k: 推薦する座席数

        Returns:
            ユーザーIDをキーとした推薦結果のディクショナリ
        """
        recommendations = {}

        for user_id in user_ids:
            try:
                recs = self.recommend_seats(user_id, features_df, top_k)
                recommendations[user_id] = recs
            except ValueError:
                print(f"Warning: User {user_id} not found")
                continue

        return recommendations

    def evaluate(
        self,
        features_df: pd.DataFrame,
        target_col: str = 'overall_rating'
    ) -> Dict[str, float]:
        """
        推薦システムを評価

        Args:
            features_df: 特徴量DataFrame（実際の評価を含む）
            target_col: ターゲット列名

        Returns:
            評価メトリクス
        """
        # 評価データがあるレコードのみ
        test_data = features_df.dropna(subset=[target_col])

        if len(test_data) == 0:
            raise ValueError("No test data with ratings found")

        # 特徴量を準備
        feature_cols = [col for col in test_data.columns
                       if col not in ['user_id', 'seat_id', 'overall_rating',
                                     'temperature_rating', 'noise_rating', 'light_rating']]

        X = test_data[feature_cols].values
        y_true = test_data[target_col].values

        # 予測
        y_pred = self.model.predict(X)

        # メトリクスを計算
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred)
        }

        return metrics


def create_recommender(model_type: str = 'random_forest', **kwargs) -> BaseSeatRecommender:
    """
    推薦モデルを作成

    Args:
        model_type: モデルタイプ ('random_forest', 'gradient_boosting', 'neural_network', 'linear')
        **kwargs: モデル固有のパラメータ

    Returns:
        推薦モデル
    """
    model_map = {
        'random_forest': RandomForestRecommender,
        'gradient_boosting': GradientBoostingRecommender,
        'neural_network': NeuralNetworkRecommender,
        'linear': LinearRecommender
    }

    if model_type not in model_map:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Choose from {list(model_map.keys())}")

    return model_map[model_type](**kwargs)
