"""
データセット処理モジュール

Hugging Face Hubからデータセットを読み込み、前処理を行うための
ユーティリティクラスを提供します。

主なクラス:
    DatasetLoader: Hugging Face Hubからデータセットを読み込む
    CustomDatasetBuilder: ローカルファイルからデータセットを作成する

使用例:
    >>> from dataset import DatasetLoader
    >>> from transformers import AutoTokenizer
    >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    >>> loader = DatasetLoader("imdb", tokenizer)
    >>> dataset = loader.load()
"""

from typing import Optional, Dict, Any
from datasets import load_dataset, Dataset, DatasetDict
from transformers import PreTrainedTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetLoader:
    """
    Hugging Face データセットのローダークラス

    Hugging Face Hubからデータセットをダウンロードし、
    トークナイズなどの前処理を行います。

    Attributes:
        dataset_name (str): データセット名
        tokenizer (PreTrainedTokenizer): トークナイザー
        max_length (int): 最大シーケンス長
        subset (str): データセットのサブセット名
    """

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
        """
        データセットを読み込む

        Hugging Face Hubからデータセットをダウンロードします。
        初回はインターネット接続が必要で、キャッシュに保存されます。

        Returns:
            DatasetDict: 読み込んだデータセット（train, test等を含む）

        Raises:
            ValueError: データセットが見つからない場合
        """
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
        # テキストカラムを自動検出
        text_columns = ["text", "sentence", "content", "review", "description", "question"]
        text_column = None
        for col in text_columns:
            if col in examples:
                text_column = col
                break

        if text_column is None:
            available_cols = list(examples.keys())
            raise ValueError(
                f"テキストカラムが見つかりません。"
                f"期待されるカラム: {text_columns}, "
                f"利用可能なカラム: {available_cols}"
            )

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
