#!/usr/bin/env python3
"""
座席性質分析ツール

座席の性質を分析し、どの性質の座席を増やすべきかを推奨します。

使用例:
    python analyze_seat_properties.py
    python analyze_seat_properties.py --data-dir data --top 5
    python analyze_seat_properties.py --seat-info data/seat_info.csv --user-profiles data/user_profiles.csv
"""

import argparse
import os
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from seat_property_analyzer import analyze_seat_properties


def main():
    parser = argparse.ArgumentParser(
        description="座席性質を分析して、どの性質の座席を増やすべきかを推奨します。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # デフォルトのデータディレクトリを使用
  python analyze_seat_properties.py

  # カスタムデータディレクトリを指定
  python analyze_seat_properties.py --data-dir my_data

  # 個別のファイルパスを指定
  python analyze_seat_properties.py --seat-info data/seat_info.csv --user-profiles data/user_profiles.csv

  # 推奨する性質の数を変更
  python analyze_seat_properties.py --top 5
        """
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="データディレクトリのパス (デフォルト: data)"
    )

    parser.add_argument(
        "--seat-info",
        type=str,
        help="座席情報CSVのパス（指定した場合はdata-dirより優先）"
    )

    parser.add_argument(
        "--user-profiles",
        type=str,
        help="ユーザープロファイルCSVのパス（指定した場合はdata-dirより優先）"
    )

    parser.add_argument(
        "--top",
        type=int,
        default=3,
        help="推奨する性質の数（デフォルト: 3）"
    )

    parser.add_argument(
        "--generate-data",
        action="store_true",
        help="データが存在しない場合、サンプルデータを生成"
    )

    args = parser.parse_args()

    # ファイルパスを決定
    if args.seat_info:
        seat_info_path = args.seat_info
    else:
        seat_info_path = os.path.join(args.data_dir, "seat_info.csv")

    if args.user_profiles:
        user_profiles_path = args.user_profiles
    else:
        user_profiles_path = os.path.join(args.data_dir, "user_profiles.csv")

    # ファイルの存在確認
    if not os.path.exists(seat_info_path) or not os.path.exists(user_profiles_path):
        if args.generate_data:
            print("データファイルが見つかりません。サンプルデータを生成します...\n")
            try:
                from data_generator import generate_all_data
                generate_all_data(
                    num_seats=50,
                    num_users=200,
                    num_timestamps=1000,
                    num_ratings=5000,
                    output_dir=args.data_dir
                )
                print("\nサンプルデータの生成が完了しました。\n")
            except ImportError:
                print("エラー: data_generatorモジュールが見つかりません。")
                print("先にデータを生成してください: python src/data_generator.py")
                sys.exit(1)
        else:
            print("エラー: 以下のファイルが見つかりません:")
            if not os.path.exists(seat_info_path):
                print(f"  - {seat_info_path}")
            if not os.path.exists(user_profiles_path):
                print(f"  - {user_profiles_path}")
            print("\nデータを生成するには --generate-data オプションを使用してください。")
            print("または、以下のコマンドでデータを生成してください:")
            print("  python src/data_generator.py")
            sys.exit(1)

    # 分析実行
    try:
        print(f"座席情報: {seat_info_path}")
        print(f"ユーザープロファイル: {user_profiles_path}")
        print(f"推奨件数: {args.top}")

        analyses, recommendations = analyze_seat_properties(
            seat_info_path,
            user_profiles_path,
            top_n=args.top
        )

        # 結果をJSONで出力するオプションを追加することも可能
        # if args.json:
        #     import json
        #     output = {
        #         'analyses': [asdict(a) for a in analyses],
        #         'recommendations': recommendations
        #     }
        #     print(json.dumps(output, ensure_ascii=False, indent=2))

    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
