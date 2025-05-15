from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch
from typing import List, Dict
import logging
from ..models.bert_classifier import load_bert_model, save_bert_model
from ..preprocessing.text_processor import prepare_dataset
from ..data.ingest import load_dataset, split_dataset

logger = logging.getLogger(__name__)

class SentimentDataset(Dataset):
    """Custom dataset for sentiment analysis."""
    def __init__(self, encodings: List[Dict]):
        self.encodings = encodings

    def __getitem__(self, idx: int) -> Dict:
        item = {
            "input_ids": self.encodings[idx]["input_ids"],
            "attention_mask": self.encodings[idx]["attention_mask"],
            "labels": torch.tensor(self.encodings[idx]["labels"], dtype=torch.long)
        }
        return item

    def __len__(self) -> int:
        return len(self.encodings)

def train_model(data_dir: str = "ml/data", model_path: str = "ml/models/sentiment_model") -> None:
    """Train and save a BERT model for sentiment analysis."""
    try:
        # Load and prepare data
        df, _ = load_dataset(data_dir)
        train_df, test_df = split_dataset(df)
        train_encodings = prepare_dataset(train_df)
        test_encodings = prepare_dataset(test_df)

        # Create datasets
        train_dataset = SentimentDataset(train_encodings)
        test_dataset = SentimentDataset(test_encodings)

        # Load model
        model = load_bert_model(model_path)

        # Define training arguments
        training_args = TrainingArguments(
            output_dir="ml/training/results",
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=50,
            weight_decay=0.01,
            logging_dir="ml/training/logs",
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True
        )

        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset
        )

        # Train model
        trainer.train()
        logger.info("Training completed successfully")

        # Save model
        save_bert_model(model, model_path)
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise