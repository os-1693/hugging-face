"""データセット処理モジュール"""

from typing import Optional, Dict, Any
from datasets import load_dataset, Dataset, DatasetDict
from transformers import PreTrainedTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetLoader:
    """Hugging Face データセットのローダークラス"""

    def __init__(
        self,
        dataset_name: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        subset: Optional[str] = None
    ):
        """
        Args:
            dataset_name: Hugging Face Hubのデータセット名
            tokenizer: トークナイザー
            max_length: 最大シーケンス長
            subset: データセットのサブセット名
        """
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.subset = subset

    def load(self) -> DatasetDict:
        """データセットを読み込む"""
        logger.info(f"Loading dataset: {self.dataset_name}")

        if self.subset:
            dataset = load_dataset(self.dataset_name, self.subset)
        else:
            dataset = load_dataset(self.dataset_name)

        logger.info(f"Dataset loaded: {dataset}")
        return dataset

    def preprocess_function(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        """
        データの前処理（テキスト分類タスク用の例）

        Args:
            examples: バッチデータ

        Returns:
            トークナイズされたデータ
        """
        # テキストカラム名はデータセットに応じて調整が必要
        text_column = "text" if "text" in examples else "sentence"

        return self.tokenizer(
            examples[text_column],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

    def prepare_dataset(
        self,
        dataset: DatasetDict,
        num_proc: int = 4,
        remove_columns: Optional[list] = None
    ) -> DatasetDict:
        """
        データセットの前処理を適用

        Args:
            dataset: 処理対象のデータセット
            num_proc: 並列処理数
            remove_columns: 削除するカラム名のリスト

        Returns:
            前処理済みデータセット
        """
        logger.info("Preprocessing dataset...")

        # デフォルトで削除するカラムを設定
        if remove_columns is None and "train" in dataset:
            remove_columns = [
                col for col in dataset["train"].column_names
                if col not in ["label", "labels"]
            ]

        processed_dataset = dataset.map(
            self.preprocess_function,
            batched=True,
            num_proc=num_proc,
            remove_columns=remove_columns,
        )

        logger.info("Dataset preprocessing completed")
        return processed_dataset


class CustomDatasetBuilder:
    """カスタムデータセットビルダー"""

    @staticmethod
    def from_csv(
        train_path: str,
        validation_path: Optional[str] = None,
        test_path: Optional[str] = None
    ) -> DatasetDict:
        """
        CSVファイルからデータセットを作成

        Args:
            train_path: 訓練データのパス
            validation_path: 検証データのパス
            test_path: テストデータのパス

        Returns:
            DatasetDict
        """
        datasets = {}

        datasets["train"] = Dataset.from_csv(train_path)

        if validation_path:
            datasets["validation"] = Dataset.from_csv(validation_path)

        if test_path:
            datasets["test"] = Dataset.from_csv(test_path)

        return DatasetDict(datasets)

    @staticmethod
    def from_json(
        train_path: str,
        validation_path: Optional[str] = None,
        test_path: Optional[str] = None
    ) -> DatasetDict:
        """
        JSONファイルからデータセットを作成

        Args:
            train_path: 訓練データのパス
            validation_path: 検証データのパス
            test_path: テストデータのパス

        Returns:
            DatasetDict
        """
        datasets = {}

        datasets["train"] = Dataset.from_json(train_path)

        if validation_path:
            datasets["validation"] = Dataset.from_json(validation_path)

        if test_path:
            datasets["test"] = Dataset.from_json(test_path)

        return DatasetDict(datasets)
