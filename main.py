import os
from src.logger import get_logger
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.utils import load_object
from src.exception import CustomException

logger = get_logger("Main")

DATA_PATH = "Telco-Customer-Churn.csv"   # <-- ensure this file is in project root or adjust path
ARTIFACTS_DIR = "artifacts"

def run_pipeline():
    try:
        logger.info("=== START PIPELINE ===")
        # 1. Ingest
        di = DataIngestion(DATA_PATH, artifacts_dir=ARTIFACTS_DIR)
        train_path, test_path = di.run()

        # 2. Transform
        dt = DataTransformation(artifacts_dir=ARTIFACTS_DIR)
        X_train, X_test, y_train, y_test, preprocessor_path, num_cols, cat_cols = dt.run(train_path, test_path)

        logger.info(f"Transformed shapes: X_train={X_train.shape}, X_test={X_test.shape}")
        logger.info(f"Preprocessor saved to {preprocessor_path}")

        # 3. Model training + tuning + selection
        mt = ModelTrainer(artifacts_dir=ARTIFACTS_DIR)
        result = mt.run(X_train, y_train, X_test, y_test, cv=4, n_iter=40)

        logger.info("=== PIPELINE COMPLETE ===")
        logger.info(f"Best model: {result['best_model_name']}")
        logger.info(f"Model and artifacts in {ARTIFACTS_DIR}")
        print(result["results_df"])
    except Exception as e:
        raise CustomException(f"Pipeline failed: {e}")

if __name__ == "__main__":
    run_pipeline()
