"""
Trains the Analytics classifier: predicts document category and priority
from cheap, immediately-available metadata (page count, file size, word
count, OCR confidence, upload hour).

Every training run is logged to MLflow (params, metrics, model artifact)
so results are comparable across runs and weeks.

Usage:
    python3 -m modules.analytics.train
"""

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split

from modules.core.logging_config import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)

FEATURE_COLUMNS = ["page_count", "file_size_kb", "word_count", "ocr_confidence", "upload_hour"]

mlflow.set_experiment("omniassist-analytics")


def load_data(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def train_classifier(df: pd.DataFrame, target_column: str) -> GridSearchCV:
    X = df[FEATURE_COLUMNS]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    with mlflow.start_run(run_name=f"analytics-{target_column}"):
        grid_search = GridSearchCV(
            estimator=RandomForestClassifier(random_state=42, class_weight="balanced"),
            param_grid=param_grid,
            cv=5,
            scoring="f1_macro",
            n_jobs=-1,
        )
        grid_search.fit(X_train, y_train)

        y_pred = grid_search.predict(X_test)
        test_f1_macro = f1_score(y_test, y_pred, average="macro")

        logger.info(f"[{target_column}] Best params: {grid_search.best_params_}")
        logger.info(f"[{target_column}] Best CV f1_macro: {grid_search.best_score_:.3f}")
        logger.info(f"[{target_column}] Test set report:\n{classification_report(y_test, y_pred)}")

        # --- MLflow logging ---
        mlflow.log_param("target", target_column)
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("cv_f1_macro", grid_search.best_score_)
        mlflow.log_metric("test_f1_macro", test_f1_macro)
        mlflow.sklearn.log_model(grid_search.best_estimator_, name=f"{target_column}_model")

    return grid_search


def main():
    df = load_data("data/sample_documents.csv")

    logger.info("Training category classifier...")
    category_model = train_classifier(df, "category")

    logger.info("Training priority classifier...")
    priority_model = train_classifier(df, "priority")

    joblib.dump(category_model.best_estimator_, "models/analytics/category_model.joblib")
    joblib.dump(priority_model.best_estimator_, "models/analytics/priority_model.joblib")
    logger.info("Saved both models to models/analytics/")


if __name__ == "__main__":
    main()