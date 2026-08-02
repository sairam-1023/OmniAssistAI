"""
Trains the Analytics classifier: predicts document category and priority
from cheap, immediately-available metadata (page count, file size, word
count, OCR confidence, upload hour).

Usage:
    python3 -m modules.analytics.train
"""

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split

from modules.core.logging_config import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)

FEATURE_COLUMNS = ["page_count", "file_size_kb", "word_count", "ocr_confidence", "upload_hour"]


def load_data(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def train_classifier(df: pd.DataFrame, target_column: str) -> GridSearchCV:
    """
    Trains a RandomForestClassifier for a single target column
    (either 'category' or 'priority'), tuned via GridSearchCV.
    """
    X = df[FEATURE_COLUMNS]
    y = df[target_column]

    # Hold back 20% of data, never seen during training, for honest evaluation.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Grid of hyperparameters to try. GridSearchCV trains one model per
    # combination and picks the best via cross-validation.
    param_grid = {
        "n_estimators": [50, 100, 200],       # number of trees in the forest
        "max_depth": [None, 5, 10],           # how deep each tree can grow
        "min_samples_leaf": [1, 2, 4],        # min samples required at a leaf node
    }

    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42, class_weight="balanced"),
        param_grid=param_grid,
        cv=5,                # 5-fold cross-validation
        scoring="f1_macro",  # better than plain accuracy for imbalanced classes
        n_jobs=-1,            # use all CPU cores
    )
    grid_search.fit(X_train, y_train)

    logger.info(f"[{target_column}] Best params: {grid_search.best_params_}")
    logger.info(f"[{target_column}] Best CV f1_macro: {grid_search.best_score_:.3f}")

    # Evaluate honestly on the held-out test set
    y_pred = grid_search.predict(X_test)
    logger.info(f"[{target_column}] Test set report:\n{classification_report(y_test, y_pred)}")

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