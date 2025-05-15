import pandas as pd
import os
from typing import Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def load_dataset(data_dir: str = "ml/data", dataset_name: str = "reviews.csv") -> Tuple[pd.DataFrame, bool]:
    """Load or create a sentiment analysis dataset from CSV or generate a sample."""
    data_path = Path(data_dir) / dataset_name
    try:
        if data_path.exists():
            df = pd.read_csv(data_path, usecols=["text", "sentiment"])
            if df.empty or "text" not in df or "sentiment" not in df:
                raise ValueError("Invalid dataset: missing required columns or empty")
            logger.info(f"Loaded dataset from {data_path} with {len(df)} samples")
            return df, True
        else:
            logger.warning(f"No dataset found at {data_path}. Generating sample data.")
            sample_data = {
                "text": [
                    "I love this product, it's amazing!",
                    "Terrible service, never buying again.",
                    "The item was okay, nothing special.",
                    "Fantastic experience, highly recommend!",
                    "Very disappointed with the quality."
                ],
                "sentiment": ["positive", "negative", "neutral", "positive", "negative"]
            }
            df = pd.DataFrame(sample_data)
            os.makedirs(data_dir, exist_ok=True)
            df.to_csv(data_path, index=False)
            logger.info(f"Created sample dataset at {data_path} with {len(df)} samples")
            return df, False
    except Exception as e:
        logger.error(f"Failed to load dataset: {str(e)}")
        raise

def split_dataset(df: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataset into train and test sets."""
    train_size = int(len(df) * train_ratio)
    train_df = df.sample(n=train_size, random_state=42)
    test_df = df.drop(train_df.index)
    logger.info(f"Split dataset: {len(train_df)} train, {len(test_df)} test samples")
    return train_df, test_df