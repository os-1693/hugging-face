"""モデル定義モジュール"""

from typing import Optional
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForQuestionAnswering,
    AutoModelForCausalLM,
    AutoConfig,
    PreTrainedModel
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelBuilder:
    """Hugging Faceモデルのビルダークラス"""

    @staticmethod
    def build_sequence_classification_model(
        model_name: str,
        num_labels: int,
        pretrained: bool = True,
        config_overrides: Optional[dict] = None
    ) -> PreTrainedModel:
        """
        テキスト分類モデルを構築

        Args:
            model_name: モデル名（例: "bert-base-uncased"）
            num_labels: ラベル数
            pretrained: 事前学習済みモデルを使用するか
            config_overrides: 設定のオーバーライド

        Returns:
            分類モデル
        """
        logger.info(f"Building sequence classification model: {model_name}")

        try:
            config = AutoConfig.from_pretrained(model_name)
        except OSError as e:
            logger.error(f"Model '{model_name}' not found")
            logger.info("Available models: https://huggingface.co/models")
            raise ValueError(f"Invalid model name: {model_name}") from e

        config.num_labels = num_labels

        if config_overrides:
            for key, value in config_overrides.items():
                setattr(config, key, value)

        try:
            if pretrained:
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    config=config,
                    ignore_mismatched_sizes=True
                )
            else:
                model = AutoModelForSequenceClassification.from_config(config)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

        logger.info(f"Model built successfully with {num_labels} labels")
        return model

    @staticmethod
    def build_token_classification_model(
        model_name: str,
        num_labels: int,
        pretrained: bool = True,
        config_overrides: Optional[dict] = None
    ) -> PreTrainedModel:
        """
        トークン分類モデルを構築（NERなど）

        Args:
            model_name: モデル名
            num_labels: ラベル数
            pretrained: 事前学習済みモデルを使用するか
            config_overrides: 設定のオーバーライド

        Returns:
            トークン分類モデル
        """
        logger.info(f"Building token classification model: {model_name}")

        try:
            config = AutoConfig.from_pretrained(model_name)
        except OSError as e:
            logger.error(f"Model '{model_name}' not found")
            logger.info("Available models: https://huggingface.co/models")
            raise ValueError(f"Invalid model name: {model_name}") from e

        config.num_labels = num_labels

        if config_overrides:
            for key, value in config_overrides.items():
                setattr(config, key, value)

        try:
            if pretrained:
                model = AutoModelForTokenClassification.from_pretrained(
                    model_name,
                    config=config,
                    ignore_mismatched_sizes=True
                )
            else:
                model = AutoModelForTokenClassification.from_config(config)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

        logger.info("Token classification model built successfully")
        return model

    @staticmethod
    def build_qa_model(
        model_name: str,
        pretrained: bool = True,
        config_overrides: Optional[dict] = None
    ) -> PreTrainedModel:
        """
        質問応答モデルを構築

        Args:
            model_name: モデル名
            pretrained: 事前学習済みモデルを使用するか
            config_overrides: 設定のオーバーライド

        Returns:
            質問応答モデル
        """
        logger.info(f"Building QA model: {model_name}")

        try:
            config = AutoConfig.from_pretrained(model_name)
        except OSError as e:
            logger.error(f"Model '{model_name}' not found")
            logger.info("Available models: https://huggingface.co/models")
            raise ValueError(f"Invalid model name: {model_name}") from e

        if config_overrides:
            for key, value in config_overrides.items():
                setattr(config, key, value)

        try:
            if pretrained:
                model = AutoModelForQuestionAnswering.from_pretrained(
                    model_name,
                    config=config,
                    ignore_mismatched_sizes=True
                )
            else:
                model = AutoModelForQuestionAnswering.from_config(config)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

        logger.info("QA model built successfully")
        return model

    @staticmethod
    def build_causal_lm_model(
        model_name: str,
        pretrained: bool = True,
        config_overrides: Optional[dict] = None
    ) -> PreTrainedModel:
        """
        因果言語モデルを構築（GPTなど）

        Args:
            model_name: モデル名
            pretrained: 事前学習済みモデルを使用するか
            config_overrides: 設定のオーバーライド

        Returns:
            因果言語モデル
        """
        logger.info(f"Building Causal LM model: {model_name}")

        try:
            config = AutoConfig.from_pretrained(model_name)
        except OSError as e:
            logger.error(f"Model '{model_name}' not found")
            logger.info("Available models: https://huggingface.co/models")
            raise ValueError(f"Invalid model name: {model_name}") from e

        if config_overrides:
            for key, value in config_overrides.items():
                setattr(config, key, value)

        try:
            if pretrained:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    config=config
                )
            else:
                model = AutoModelForCausalLM.from_config(config)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

        logger.info("Causal LM model built successfully")
        return model
