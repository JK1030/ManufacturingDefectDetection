"""
Train a Gradient Boosting model for manufacturing defect detection.

This model loads featured manufacturing data, splits it into training 
and testing sets, trains a Gradient Boosting classifier, saves the trained 
model, and displays feature importance.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split


def load_data(input_file: str) -> pd.DataFrame:
    """
    Load featured manufactuaring data from a CSV file. 
    """
    return pd.read_csv(input_file)

def split_features_target(
        df: pd.DataFrame, target_column: str
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split dataframe into features and targets.
    """
    X = df.drop(columns=[target_column, "lot_id"], errors="ignore")
    y = df[target_column]

    return X,y

def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split features and target into training and testing sets. 
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> GradientBoostingClassifier:
    """
    Train a Gradient Boosting classifier
    """
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)

    return model

def save_model(model: GradientBoostingClassifier, output_file:str) -> None:
    """
    Save trained model to a file
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, output_path)

def show_feature_importance(
        model: GradientBoostingClassifier,
        feature_names: list[str]
) -> pd.DataFrame:
    """
    Create and print feature importance dataframe
    """
    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_
        }
        ).sort_values(by="importance", ascending=False)

    print("\nFeature Importance:")
    print(importance_df)
    
    return importance_df

def main() -> None:
    """
    Run model training pipline
    """
    input_file = "data/featured/featured_synthetic_manufacturing_data.csv"
    output_model_file = "models/gradient_boosting_model.pkl"

    df = load_data(input_file)

    X,y = split_features_target(df, target_column="defect")
    X_train, X_test, y_train, y_test = split_train_test(X,y)

    model = train_model(X_train, y_train)

    save_model(model, output_model_file)

    show_feature_importance(model, X.columns.tolist())

    print("\nModel training completed.")
    print(f"Train rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")
    print(f"Model saved to: {output_model_file}")


if __name__ == "__main__":
    main()