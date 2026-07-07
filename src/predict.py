"""
Prediction pipeline for manufacturing defect detection.
"""
import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

def load_model(model_path: str) -> GradientBoostingClassifier:
    """
    Load trained model from file.
    """
    return joblib.load(model_path)

def load_data(input_path: str) -> pd.DataFrame:
    """
    Load new manufacturing data.
    """
    return pd.read_csv(input_path)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the same preprocessing used during training.
    """
    # TODO: resuse data_cleaning + feature_engineering logic
    return df

def make_prediction(model, X:pd.DataFrame) -> pd.DataFrame:
    """
    Generate defect predictions.
    """
    predictions = model.predict(X)

    result_df = X.copy()
    result_df["predicted_defect"] = predictions

    return result_df

def save_prediction(result_df: pd.DataFrame, output_path: str) -> None:
    """
    Save prediction results to CSV,
    """
    result_df.to_csv(output_path, index=False)


def main() -> None:
    model_path = "models/gradient_boosting_model.pkl"
    input_path = "data/featured/featured_synthetic_manufacturing_data.csv"
    output_path = "reports/prediction_result.csv"

    model = load_model(model_path)
    df = load_data(input_path)
    X = df.drop(columns=["defect", "lot_id"])
    result_df = make_prediction(model, X)

    save_prediction(result_df, output_path)

    print("Prediction completed")
    print(result_df.head())

if __name__=="__main__":
    main()
