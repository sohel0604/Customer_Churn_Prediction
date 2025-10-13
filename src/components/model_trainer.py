import os
import sys
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from src.logger import get_logger
from src.utils import save_object, save_df, save_json
from src.exception import CustomException

logger = get_logger("ModelTrainer")

class ModelTrainer:
    def __init__(self, artifacts_dir="artifacts", random_state=42):
        self.artifacts_dir = artifacts_dir
        os.makedirs(self.artifacts_dir, exist_ok=True)
        self.best_model_path = os.path.join(self.artifacts_dir, "best_model.pkl")
        self.results_path = os.path.join(self.artifacts_dir, "model_comparison.csv")
        self.search_reports = os.path.join(self.artifacts_dir, "search_reports.json")
        self.random_state = random_state

    def _get_models_and_params(self):
        """Return models and hyperparameter grids"""
        models = {
            "logreg": LogisticRegression(max_iter=1000, random_state=self.random_state, solver="saga"),
            "rf": RandomForestClassifier(random_state=self.random_state),
            "gb": GradientBoostingClassifier(random_state=self.random_state),
            "xgb": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=self.random_state),
            "svc": SVC(probability=True, random_state=self.random_state),
        }

        params = {
            "logreg": {
                "C": np.logspace(-4, 4, 20),
                "penalty": ["l1", "l2", "elasticnet"],
                "l1_ratio": [0, 0.5, 1],
            },
            "rf": {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 5, 10, 20],
                "min_samples_split": [2, 5, 10],
            },
            "gb": {
                "n_estimators": [100, 200],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [3, 5, 8],
            },
            "xgb": {
                "n_estimators": [100, 200],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [3, 5, 8],
                "subsample": [0.6, 0.8, 1.0],
            },
            "svc": {
                "C": [0.1, 1, 10],
                "kernel": ["rbf", "linear"],
                "gamma": ["scale", "auto"],
            }
        }

        return models, params

    def run_search(self, model, param_dist, X, y, n_iter=30, cv=5, scoring="f1", n_jobs=-1):
        """RandomizedSearchCV with timing and logging"""
        try:
            cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_dist,
                n_iter=n_iter,
                scoring=scoring,
                n_jobs=n_jobs,
                cv=cv_strategy,
                verbose=0,
                random_state=self.random_state,
            )
            t0 = time.time()
            search.fit(X, y)
            t1 = time.time()
            runtime = round(t1 - t0, 2)
            return search, runtime
        except Exception as e:
            raise CustomException(e, sys)

    def evaluate(self, model, X_test, y_test):
        """Compute test metrics"""
        preds = model.predict(X_test)
        metrics = {
            "accuracy": float(accuracy_score(y_test, preds)),
            "f1": float(f1_score(y_test, preds)),
            "precision": float(precision_score(y_test, preds)),
            "recall": float(recall_score(y_test, preds)),
            "report": classification_report(y_test, preds)
        }
        return metrics

    def run(self, X_train, y_train, X_test, y_test, cv=4, n_iter=40):
        try:
            
            # Step 1: Apply SMOTE
            
            logger.info("Applying SMOTE to balance the dataset...")
            smote = SMOTE(random_state=self.random_state)
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
            logger.info(f"Before SMOTE: {np.bincount(y_train)} | After SMOTE: {np.bincount(y_train_res)}")

            models, params = self._get_models_and_params()
            results = []
            search_reports = {}
            trained_models = {}

            
            # Step 2: Model Training + Tuning
            
            for name, model in models.items():
                logger.info(f"Starting training for model: {name}")
                param_dist = params.get(name, {})

                if param_dist:
                    search, runtime = self.run_search(model, param_dist, X_train_res, y_train_res, n_iter=min(n_iter, 40), cv=cv)
                    best_est = search.best_estimator_
                    best_params = search.best_params_
                    best_score = search.best_score_
                    logger.info(f"{name} best_score={best_score}, params={best_params}")
                else:
                    model.fit(X_train_res, y_train_res)
                    best_est = model
                    best_params = {}
                    best_score = None
                    runtime = 0

                trained_models[name] = best_est
                metrics = self.evaluate(best_est, X_test, y_test)

                results.append({
                    "model": name,
                    "test_accuracy": metrics["accuracy"],
                    "test_f1": metrics["f1"],
                    "test_precision": metrics["precision"],
                    "test_recall": metrics["recall"],
                    "best_params": best_params,
                    "cv_best_score": best_score
                })

                search_reports[name] = {
                    "best_score": best_score,
                    "best_params": best_params,
                    "runtime_sec": runtime
                }

            
            #  Step 3: Save Results
            
            results_df = pd.DataFrame(results).sort_values(by="test_f1", ascending=False).reset_index(drop=True)
            save_df(results_df, self.results_path)
            save_object(results_df, os.path.join(self.artifacts_dir, "model_results_df.pkl"))
            save_json(search_reports, self.search_reports)
            logger.info(f"Saved model comparison to {self.results_path}")

            
            #  Step 4: Save Best Model
            
            best_model_name = results_df.iloc[0]["model"]
            best_model = trained_models[best_model_name]
            save_object(best_model, self.best_model_path)
            logger.info(f" Saved best model '{best_model_name}' to {self.best_model_path}")

            return {
                "best_model_name": best_model_name,
                "best_model_path": self.best_model_path,
                "results_df_path": self.results_path,
                "results_df": results_df
            }

        except Exception as e:
            raise CustomException(f"Model training/selection failed: {e}")
