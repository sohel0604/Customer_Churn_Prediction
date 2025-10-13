import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.utils import save_object
from src.logger import get_logger
from src.exception import CustomException

logger = get_logger("DataTransformation")

class DataTransformation:
    def __init__(self, artifacts_dir: str = "artifacts"):
        self.artifacts_dir = artifacts_dir
        os.makedirs(self.artifacts_dir, exist_ok=True)
        self.preprocessor_path = os.path.join(self.artifacts_dir, "preprocessor.pkl")

    def _clean_telco(self, df: pd.DataFrame) -> pd.DataFrame:
        # convert TotalCharges to numeric (Telco dataset has blanks)
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        # strip spaces in object dtype columns
        obj_cols = df.select_dtypes(include=["object"]).columns
        for c in obj_cols:
            df[c] = df[c].str.strip()
        return df

    def get_preprocessor(self, df: pd.DataFrame):
        try:
            df = df.copy()
            df = self._clean_telco(df)
            # drop id if exists
            if "customerID" in df.columns:
                df = df.drop(columns=["customerID"])

            # identify numeric and categorical
            numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
            # remove target if present
            if "Churn" in numeric_cols:
                numeric_cols.remove("Churn")
            # Force common numeric columns to numeric
            for c in ["tenure", "MonthlyCharges", "TotalCharges"]:
                if c in df.columns and c not in numeric_cols:
                    numeric_cols.append(c)

            categorical_cols = [c for c in df.columns if c not in numeric_cols and c != "Churn"]

            logger.info(f"Numeric cols: {numeric_cols}")
            logger.info(f"Categorical cols: {categorical_cols}")

            # numeric pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            # categorical pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", num_pipeline, numeric_cols),
                    ("cat", cat_pipeline, categorical_cols),
                ],
                remainder="drop",
            )

            return preprocessor, numeric_cols, categorical_cols
        except Exception as e:
            raise CustomException(f"Error building preprocessor: {e}")

    def run(self, train_path: str, test_path: str, target_col: str = "Churn"):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            train_df = self._clean_telco(train_df)
            test_df = self._clean_telco(test_df)

            # Drop id
            drop_cols = [c for c in ["customerID"] if c in train_df.columns]
            if drop_cols:
                train_df = train_df.drop(columns=drop_cols)
                test_df = test_df.drop(columns=drop_cols)

            # Map target to 0/1
            if target_col in train_df.columns:
                y_train = train_df[target_col].map({"Yes": 1, "No": 0}).values
                X_train = train_df.drop(columns=[target_col])
                y_test = test_df[target_col].map({"Yes": 1, "No": 0}).values
                X_test = test_df.drop(columns=[target_col])
            else:
                raise CustomException(f"Target column '{target_col}' not found in train data")

            preprocessor, numeric_cols, categorical_cols = self.get_preprocessor(train_df)

            # Fit transform
            X_train_trans = preprocessor.fit_transform(X_train)
            X_test_trans = preprocessor.transform(X_test)

            save_object(preprocessor, self.preprocessor_path)
            logger.info(f"Saved preprocessor to {self.preprocessor_path}")

            return X_train_trans, X_test_trans, y_train, y_test, self.preprocessor_path, numeric_cols, categorical_cols
        except Exception as e:
            raise CustomException(f"Data transformation failed: {e}")
