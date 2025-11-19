"""
座席性質分析モジュール

座席の性質を分析し、どの性質の座席を増やすべきかを推奨します。
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class PropertyAnalysis:
    """座席性質の分析結果"""
    property_name: str
    property_name_ja: str
    current_supply: int
    supply_ratio: float
    user_demand: int
    demand_ratio: float
    gap: int
    gap_ratio: float
    shortage_score: float
    recommendation_priority: int


class SeatPropertyAnalyzer:
    """座席性質分析クラス"""

    # 座席性質の日本語名マッピング
    PROPERTY_NAMES = {
        'is_window': '窓際座席',
        'near_ac': '空調近く',
        'near_entrance': '出入口近く',
        'quiet_area': '静かなエリア',
        'has_monitor': 'モニター付き',
        'has_standing_desk': 'スタンディングデスク'
    }

    # ユーザー好みと座席性質のマッピング
    USER_PREFERENCE_MAPPING = {
        'prefers_window': 'is_window',
        'prefers_quiet': 'quiet_area',
        'needs_monitor': 'has_monitor',
        'prefers_standing_desk': 'has_standing_desk'
    }

    def __init__(self, seat_info: pd.DataFrame, user_profiles: pd.DataFrame):
        """
        Args:
            seat_info: 座席情報DataFrame
            user_profiles: ユーザープロファイルDataFrame
        """
        self.seat_info = seat_info
        self.user_profiles = user_profiles
        self.total_seats = len(seat_info)
        self.total_users = len(user_profiles)

    def analyze_property_supply(self) -> Dict[str, Tuple[int, float]]:
        """
        座席性質ごとの供給を分析

        Returns:
            性質ごとの(座席数, 割合)の辞書
        """
        supply = {}

        for prop in self.PROPERTY_NAMES.keys():
            if prop in self.seat_info.columns:
                count = int(self.seat_info[prop].sum())
                ratio = count / self.total_seats if self.total_seats > 0 else 0
                supply[prop] = (count, ratio)
            else:
                supply[prop] = (0, 0.0)

        return supply

    def analyze_property_demand(self) -> Dict[str, Tuple[int, float]]:
        """
        座席性質ごとの需要を分析

        Returns:
            性質ごとの(ユーザー数, 割合)の辞書
        """
        demand = {}

        # ユーザー好みから座席性質への需要を集計
        for user_pref, seat_prop in self.USER_PREFERENCE_MAPPING.items():
            if user_pref in self.user_profiles.columns:
                count = int(self.user_profiles[user_pref].sum())
                ratio = count / self.total_users if self.total_users > 0 else 0
                demand[seat_prop] = (count, ratio)
            else:
                demand[seat_prop] = (0, 0.0)

        # near_acとnear_entranceは直接的なユーザー好みがないため、
        # 温度敏感性や静かさの好みから推定
        if 'user_type' in self.user_profiles.columns:
            # 温度敏感ユーザーは空調近くを好む傾向
            temp_sensitive = self.user_profiles[
                self.user_profiles['user_type'].isin(['cold_sensitive', 'heat_sensitive'])
            ]
            count = len(temp_sensitive)
            ratio = count / self.total_users if self.total_users > 0 else 0
            demand['near_ac'] = (count, ratio)

            # 出入口近くは一般的に避けられる（需要低い）
            # 全ユーザーの10%程度が気にしないと仮定
            count = int(self.total_users * 0.1)
            ratio = 0.1
            demand['near_entrance'] = (count, ratio)
        else:
            demand['near_ac'] = (0, 0.0)
            demand['near_entrance'] = (0, 0.0)

        return demand

    def calculate_shortage_score(
        self,
        supply_ratio: float,
        demand_ratio: float
    ) -> float:
        """
        不足度スコアを計算

        Args:
            supply_ratio: 供給割合
            demand_ratio: 需要割合

        Returns:
            不足度スコア (0-100、高いほど不足)
        """
        if demand_ratio == 0:
            return 0.0

        # 需要に対する供給の不足度
        shortage = max(0, demand_ratio - supply_ratio)

        # 需要の大きさも考慮（需要が大きいほど重要）
        shortage_score = (shortage / demand_ratio) * 100 * demand_ratio

        return min(100.0, shortage_score)

    def analyze_all_properties(self) -> List[PropertyAnalysis]:
        """
        全ての座席性質を分析

        Returns:
            PropertyAnalysisのリスト（不足度の高い順）
        """
        supply = self.analyze_property_supply()
        demand = self.analyze_property_demand()

        analyses = []

        for prop in self.PROPERTY_NAMES.keys():
            supply_count, supply_ratio = supply.get(prop, (0, 0.0))
            demand_count, demand_ratio = demand.get(prop, (0, 0.0))

            gap = demand_count - supply_count
            gap_ratio = demand_ratio - supply_ratio

            shortage_score = self.calculate_shortage_score(supply_ratio, demand_ratio)

            analysis = PropertyAnalysis(
                property_name=prop,
                property_name_ja=self.PROPERTY_NAMES[prop],
                current_supply=supply_count,
                supply_ratio=supply_ratio,
                user_demand=demand_count,
                demand_ratio=demand_ratio,
                gap=gap,
                gap_ratio=gap_ratio,
                shortage_score=shortage_score,
                recommendation_priority=0  # 後で設定
            )

            analyses.append(analysis)

        # 不足度スコアでソート
        analyses.sort(key=lambda x: x.shortage_score, reverse=True)

        # 優先順位を設定
        for i, analysis in enumerate(analyses, 1):
            analysis.recommendation_priority = i

        return analyses

    def generate_recommendations(
        self,
        analyses: List[PropertyAnalysis],
        top_n: int = 3
    ) -> List[Dict]:
        """
        座席増設の推奨を生成

        Args:
            analyses: PropertyAnalysisのリスト
            top_n: 上位N件を推奨

        Returns:
            推奨のリスト
        """
        recommendations = []

        for analysis in analyses[:top_n]:
            if analysis.shortage_score > 5:  # 閾値以上の不足がある場合
                # 推奨増設数を計算（需要を満たすために必要な数）
                recommended_addition = max(1, int(analysis.gap * 0.7))  # 70%のギャップを埋める

                recommendation = {
                    'priority': analysis.recommendation_priority,
                    'property': analysis.property_name_ja,
                    'property_en': analysis.property_name,
                    'reason': self._generate_reason(analysis),
                    'current_count': analysis.current_supply,
                    'demand_count': analysis.user_demand,
                    'shortage_score': round(analysis.shortage_score, 2),
                    'recommended_addition': recommended_addition,
                    'target_count': analysis.current_supply + recommended_addition
                }

                recommendations.append(recommendation)

        return recommendations

    def _generate_reason(self, analysis: PropertyAnalysis) -> str:
        """
        推奨理由を生成

        Args:
            analysis: PropertyAnalysis

        Returns:
            推奨理由の文字列
        """
        supply_pct = analysis.supply_ratio * 100
        demand_pct = analysis.demand_ratio * 100

        if analysis.gap > 0:
            return (
                f"需要{demand_pct:.1f}%に対して供給{supply_pct:.1f}%と不足しています。"
                f"{analysis.gap}席分の需要が満たされていません。"
            )
        else:
            return (
                f"需要{demand_pct:.1f}%に対して供給{supply_pct:.1f}%と十分です。"
            )

    def print_analysis_report(self, analyses: List[PropertyAnalysis]):
        """
        分析レポートを表示

        Args:
            analyses: PropertyAnalysisのリスト
        """
        print("\n" + "=" * 80)
        print("座席性質分析レポート")
        print("=" * 80)
        print(f"\n総座席数: {self.total_seats}")
        print(f"総ユーザー数: {self.total_users}")

        print("\n" + "-" * 80)
        print("座席性質ごとの需要供給分析")
        print("-" * 80)

        # テーブルヘッダー
        print(f"{'性質':<15} {'供給':<12} {'需要':<12} {'ギャップ':<10} {'不足度':<8} {'優先度':<6}")
        print("-" * 80)

        for analysis in analyses:
            supply_str = f"{analysis.current_supply}席 ({analysis.supply_ratio*100:.1f}%)"
            demand_str = f"{analysis.user_demand}人 ({analysis.demand_ratio*100:.1f}%)"
            gap_str = f"{analysis.gap:+d}席"
            shortage_str = f"{analysis.shortage_score:.1f}"
            priority_str = f"{analysis.recommendation_priority}"

            print(f"{analysis.property_name_ja:<15} {supply_str:<12} {demand_str:<12} "
                  f"{gap_str:<10} {shortage_str:<8} {priority_str:<6}")

        print("-" * 80)

    def print_recommendations(self, recommendations: List[Dict]):
        """
        推奨事項を表示

        Args:
            recommendations: 推奨のリスト
        """
        print("\n" + "=" * 80)
        print("座席増設の推奨")
        print("=" * 80)

        if not recommendations:
            print("\n現在の座席配置は需要を十分に満たしています。")
            return

        for rec in recommendations:
            print(f"\n【優先度 {rec['priority']}】 {rec['property']}")
            print(f"  理由: {rec['reason']}")
            print(f"  現在: {rec['current_count']}席")
            print(f"  需要: {rec['demand_count']}人")
            print(f"  不足度スコア: {rec['shortage_score']}")
            print(f"  推奨増設数: {rec['recommended_addition']}席")
            print(f"  目標: {rec['target_count']}席")

        print("\n" + "=" * 80)


def analyze_seat_properties(
    seat_info_path: str,
    user_profiles_path: str,
    top_n: int = 3
) -> Tuple[List[PropertyAnalysis], List[Dict]]:
    """
    座席性質を分析して推奨を生成

    Args:
        seat_info_path: 座席情報CSVのパス
        user_profiles_path: ユーザープロファイルCSVのパス
        top_n: 上位N件を推奨

    Returns:
        (分析結果リスト, 推奨リスト)のタプル
    """
    # データ読み込み
    seat_info = pd.read_csv(seat_info_path)
    user_profiles = pd.read_csv(user_profiles_path)

    # 分析器作成
    analyzer = SeatPropertyAnalyzer(seat_info, user_profiles)

    # 分析実行
    analyses = analyzer.analyze_all_properties()
    recommendations = analyzer.generate_recommendations(analyses, top_n)

    # レポート表示
    analyzer.print_analysis_report(analyses)
    analyzer.print_recommendations(recommendations)

    return analyses, recommendations


if __name__ == "__main__":
    # サンプル実行
    import sys

    seat_info_path = "data/seat_info.csv"
    user_profiles_path = "data/user_profiles.csv"

    if len(sys.argv) > 1:
        seat_info_path = sys.argv[1]
    if len(sys.argv) > 2:
        user_profiles_path = sys.argv[2]

    try:
        analyze_seat_properties(seat_info_path, user_profiles_path)
    except FileNotFoundError as e:
        print(f"エラー: ファイルが見つかりません - {e}")
        print(f"\n使用法: python {sys.argv[0]} [seat_info.csv] [user_profiles.csv]")
        sys.exit(1)
