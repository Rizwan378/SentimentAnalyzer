from transformers import Trainer
from torch.utils.data import Dataset
from typing import Dict
import logging
from ..models.bert_classifier import load_bert_model
from ..preprocessing.text_processor import prepare_dataset
from ..data.ingest import load_dataset
from .train import SentimentDataset

logger = logging.getLogger(__name__)

def evaluate_model(data_dir: str = "ml/data", model_path: str = "ml/models/sentiment_model") -> Dict:
    """Evaluate the trained BERT model on test data."""
    try:
        # Load and prepare test data
        df, _ = load_dataset(data_dir)
        _, test_df = split_dataset(df)
        test_encodings = prepare_dataset(test_df)
        test_dataset = SentimentDataset(test_encodings)

        # Load model
        model = load_bert_model(model_path)

        # Initialize trainer for evaluation
        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir="ml/training/results",
                per_device_eval_batch_size=8
            ),
            eval_dataset=test_dataset
        )

        # Evaluate model
        metrics = trainer.evaluate()
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise