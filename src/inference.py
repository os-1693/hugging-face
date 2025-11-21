"""推論スクリプト"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelInference:
    """モデル推論クラス"""

    def __init__(self, model_path: str, device: str = None):
        """
        Args:
            model_path: モデルのパス
            device: 使用するデバイス（cuda/cpu）
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading model from {model_path}")
        logger.info(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        # 推論最適化: キャッシュを有効化
        if hasattr(self.model, "config"):
            self.model.config.use_cache = True

    def predict(self, text: str) -> dict:
        """
        テキストの予測を行う

        Args:
            text: 入力テキスト

        Returns:
            予測結果の辞書
        """
        # トークナイズ
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )

        # デバイスに転送
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        # 推論
        with torch.no_grad():
            outputs = self.model(**inputs)

        # 結果の処理
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0][predicted_class].item()

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": probabilities[0].cpu().numpy().tolist()
        }

    def predict_batch(self, texts: list) -> list:
        """
        複数テキストの予測を行う

        Args:
            texts: 入力テキストのリスト

        Returns:
            予測結果のリスト
        """
        # トークナイズ
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )

        # デバイスに転送
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        # 推論
        with torch.no_grad():
            outputs = self.model(**inputs)

        # 結果の処理
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        predicted_classes = torch.argmax(probabilities, dim=-1).cpu().numpy().tolist()
        confidences = [probabilities[i][pred].item() for i, pred in enumerate(predicted_classes)]

        results = []
        for i, (pred_class, confidence) in enumerate(zip(predicted_classes, confidences)):
            results.append({
                "text": texts[i],
                "predicted_class": pred_class,
                "confidence": confidence,
                "probabilities": probabilities[i].cpu().numpy().tolist()
            })

        return results


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='モデル推論')
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='モデルのパス'
    )
    parser.add_argument(
        '--text',
        type=str,
        help='推論するテキスト'
    )
    parser.add_argument(
        '--texts',
        nargs='+',
        help='推論する複数のテキスト'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='使用するデバイス（cuda/cpu）'
    )

    args = parser.parse_args()

    # 推論クラスの初期化
    inference = ModelInference(args.model_path, args.device)

    # 推論の実行
    if args.text:
        result = inference.predict(args.text)
        logger.info(f"Input: {args.text}")
        logger.info(f"Predicted class: {result['predicted_class']}")
        logger.info(f"Confidence: {result['confidence']:.4f}")
        logger.info(f"Probabilities: {result['probabilities']}")

    elif args.texts:
        results = inference.predict_batch(args.texts)
        for result in results:
            logger.info("-" * 50)
            logger.info(f"Input: {result['text']}")
            logger.info(f"Predicted class: {result['predicted_class']}")
            logger.info(f"Confidence: {result['confidence']:.4f}")

    else:
        logger.error("--text または --texts を指定してください")


if __name__ == '__main__':
    main()
