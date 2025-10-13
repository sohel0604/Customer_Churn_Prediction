import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.logger import get_logger
from src.exception import CustomException
from src.utils import save_df

logger = get_logger("DataIngestion")

class DataIngestion:
    def __init__(self, input_path: str, artifacts_dir: str = "artifacts"):
        self.input_path = input_path
        self.artifacts_dir = artifacts_dir
        os.makedirs(self.artifacts_dir, exist_ok=True)
        self.raw_path = os.path.join(self.artifacts_dir, "raw.csv")
        self.train_path = os.path.join(self.artifacts_dir, "train.csv")
        self.test_path = os.path.join(self.artifacts_dir, "test.csv")

    def run(self, test_size: float = 0.2, random_state: int = 42):
        try:
            logger.info(f"Loading data from {self.input_path}")
            df = pd.read_csv(self.input_path)
            # Save raw
            save_df(df, self.raw_path)
            logger.info(f"Saved raw data to {self.raw_path}")

            # If Churn present use stratify
            stratify_col = df["Churn"] if "Churn" in df.columns else None
            train_df, test_df = train_test_split(
                df, test_size=test_size, random_state=random_state, stratify=stratify_col
            )
            save_df(train_df, self.train_path)
            save_df(test_df, self.test_path)
            logger.info(f"Saved train to {self.train_path} and test to {self.test_path}")
            return self.train_path, self.test_path
        except Exception as e:
            raise CustomException(f"Data ingestion failed: {e}")
