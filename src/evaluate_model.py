"""
Evaluate a trained Gradient Boosting model for manufacturing defect detection.

This module loads the engineered dataset and the trained model,
generates predictions on the test set, and evaluates model
performance using common classification metrics.
"""
import joblib
import pandas as pd
import time

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import(
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

def load_data(input_file: str) -> pd.DataFrame: 
    """
    Load featured manufacturing dataset.

    Args:
        input_file: Path to the input CSV file.

    Returns:
        Loaded pandas DataFrame.
    """
    return pd.read_csv(input_file) 
    
def split_features_target(
        df: pd.DataFrame, target_column:str
        ) ->tuple[pd.DataFrame, pd.Series]:
    """
    Split dataset into features and target.

    Args:
        df: Input DataFrame.
        target_column: Name of the target column.

    Returns:
        Feature DataFrame and target Series.
    """  
    X = df.drop(columns=[target_column, "lot_id"], errors="ignore")
    y = df[target_column]
    
    return X,y

def split_train_test(
        X: pd.DataFrame,
        y: pd.Series,
        test_size = 0.2,
        random_state = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split features and target into training and testing sets.

    Args:
        X: Feature DataFrame.
        y: Target Series.
        test_size: Proportion of the test dataset.
        random_state: Random seed for reproducibility.

    Returns:
        Training and testing datasets.
    """
    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

def load_model(model_file: str) -> GradientBoostingClassifier:
    """
    Load a trained machine learning model.

    Args:
        model_file: Path to the saved model file.

    Returns:
        Loaded machine learning model.
    """
    return joblib.load(model_file)

def make_prediction(model, X_test: pd.DataFrame) -> pd.Series:
    """
    Generate predictions using the trained model.

    Args:
        model: Trained machine learning model.
        X_test: Test feature DataFrame.

    Returns:
        Predicted labels.
    """
    y_pred = model.predict(X_test)
    return y_pred

def main() -> None:
    """
    Load data and trained model, then evaluate model performance.
    """   
    input_file = "data/featured/featured_synthetic_manufacturing_data.csv"
    model_file = "models/gradient_boosting_model.pkl"
    
    df = load_data(input_file)
    X, y = split_features_target(df, target_column="defect")
    X_train, X_test, y_train, y_test = split_train_test(X,y)

    model = load_model(model_file)

    start_time = time.perf_counter()

    y_pred = make_prediction(model, X_test)

    end_time = time.perf_counter()

    inference_time = end_time - start_time
    avg_latency_ms = (inference_time / len(X_test)) * 1000
    throughput = len(X_test) / inference_time


    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))

    print("\nInference Performance:")
    print(f"Test records: {len(X_test)}")
    print(f"Total inference time: {inference_time:.6f} seconds")
    print(f"Average latency: {avg_latency_ms:.6f} ms/record")
    print(f"Throughput: {throughput:.2f} records/second")


    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


if __name__ =="__main__":
    main()