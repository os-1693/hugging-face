"""モデル学習スクリプト"""

import argparse
import logging
import os
from typing import Any, Dict, Optional

import evaluate
import numpy as np
import torch
import yaml
from transformers import (
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from dataset import DatasetLoader
from model import ModelBuilder

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    設定ファイルを読み込む

    Args:
        config_path: 設定ファイルのパス

    Returns:
        設定辞書
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def compute_metrics_classification(eval_pred):
    """
    分類タスクのメトリクスを計算

    Args:
        eval_pred: 予測結果

    Returns:
        メトリクス辞書
    """
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def train(config: Dict[str, Any]):
    """
    モデルの学習を実行

    Args:
        config: 設定辞書
    """
    # シードの設定
    set_seed(config.get("seed", 42))

    # デバイスの設定
    device = torch.device("cpu")  # CPUを使用
    logger.info(f"Using device: {device}")

    # トークナイザーの読み込み
    logger.info(f"Loading tokenizer: {config['model']['name']}")
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])

    # データセットの準備
    dataset_config = config["dataset"]
    dataset_loader = DatasetLoader(
        dataset_name=dataset_config["name"],
        tokenizer=tokenizer,
        max_length=config["model"].get("max_length", 512),
        subset=dataset_config.get("subset"),
    )

    # データセットの読み込みと前処理
    raw_dataset = dataset_loader.load()
    processed_dataset = dataset_loader.prepare_dataset(
        raw_dataset, num_proc=config.get("num_proc", 4)
    )

    # モデルの構築
    model_config = config["model"]
    task_type = model_config.get("task_type", "sequence_classification")

    if task_type == "sequence_classification":
        model = ModelBuilder.build_sequence_classification_model(
            model_name=model_config["name"],
            num_labels=model_config["num_labels"],
            pretrained=model_config.get("pretrained", True),
            config_overrides=model_config.get("config_overrides"),
        )
    elif task_type == "token_classification":
        model = ModelBuilder.build_token_classification_model(
            model_name=model_config["name"],
            num_labels=model_config["num_labels"],
            pretrained=model_config.get("pretrained", True),
            config_overrides=model_config.get("config_overrides"),
        )
    elif task_type == "qa":
        model = ModelBuilder.build_qa_model(
            model_name=model_config["name"],
            pretrained=model_config.get("pretrained", True),
            config_overrides=model_config.get("config_overrides"),
        )
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    model.to(device)

    # 学習設定
    training_config = config["training"]
    training_args = TrainingArguments(
        output_dir=training_config["output_dir"],
        num_train_epochs=training_config["num_epochs"],
        per_device_train_batch_size=training_config["batch_size"],
        per_device_eval_batch_size=training_config["eval_batch_size"],
        learning_rate=training_config["learning_rate"],
        weight_decay=training_config.get("weight_decay", 0.01),
        warmup_steps=training_config.get("warmup_steps", 500),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir=training_config.get("logging_dir", "./logs"),
        logging_steps=training_config.get("logging_steps", 100),
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=training_config.get("save_total_limit", 3),
        fp16=False,  # CPU使用のためfp16無効
        report_to=training_config.get("report_to", ["tensorboard"]),
        seed=config.get("seed", 42),
    )

    # コールバックの設定
    callbacks = []
    if training_config.get("early_stopping"):
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=training_config.get(
                    "early_stopping_patience", 3
                )
            )
        )

    # Trainerの初期化
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset["train"],
        eval_dataset=processed_dataset.get("validation", processed_dataset.get("test")),
        processing_class=tokenizer,
        compute_metrics=compute_metrics_classification,
        callbacks=callbacks,
    )

    # 学習の開始
    logger.info("Starting training...")
    train_result = trainer.train()

    # 学習結果の保存
    logger.info("Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(training_config["output_dir"])

    # メトリクスの保存
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # 評価
    if "test" in processed_dataset:
        logger.info("Evaluating on test set...")
        test_metrics = trainer.evaluate(processed_dataset["test"])
        trainer.log_metrics("test", test_metrics)
        trainer.save_metrics("test", test_metrics)

    logger.info("Training completed!")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="Hugging Face モデル学習")
    parser.add_argument(
        "--config",
        type=str,
        default="config/train_config.yaml",
        help="設定ファイルのパス",
    )
    args = parser.parse_args()

    # 設定の読み込み
    config = load_config(args.config)

    # 学習の実行
    train(config)


if __name__ == "__main__":
    main()
