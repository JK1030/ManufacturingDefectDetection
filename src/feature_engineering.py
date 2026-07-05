'''
Feature Engineering pipeline for the Manufacturing Defect Detection project. 

This module loads the cleaned manufacturing dataset, creates additional 
model-ready features, encodes categorical variables, and saves the final 
featured dataset for machine learning model training. 
'''

from pathlib import Path
import pandas as pd

def load_data(input_file: str) -> pd.DataFrame:
    """
    Load the cleaned manufacturing dataset from a CSV file.

    Args:
        input_file: Path to the cleaned input CSV file.

    Returns:
        A pandas DataFrame containing the cleaned manufacturing data.
    """

    df = pd.read_csv(input_file)
    return df

def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction features from existing numerical process variables.

    The temperature-pressure interaction captures the combined effect of 
    temperature and pressure on the manufacturing process.

    Args:
        df: Input DataFrame containing temperature and pressure columns.

    Returns:
        A copy of the DataFrame with interaction features added. 
    """
    df = df.copy()
    df["temp_press_interaction"] = (df["temperature"] * df["pressure"])
    return df

def create_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create ratio-based features from numerical process variables. 
    Ratio features can help the model capture relative relationships between 
    process variables, such as temperature compared with pressure.

    Args: 
        df: Input DataFrame containing temperature and pressure columns.

    Returns:
        A copy of the DataFrame with ratio features added. 

    """
    df = df.copy()
    df["temp_press_ratio"] = (df["temperature"] / df["pressure"])
    return df

def create_risk_flags(df: pd.DataFrame) -> pd.DataFrame: 
    """
    Create binary risk flag features based on process thresholds.

    Each risk flag indicates whether a process variables is outside a selected
    warning range. These flags provide simple engineered signals that may help 
    the model identify risky manufacturing conditions.

    Args:
        df: Input DataFrame containing process variable columns.

    Returns:
        A copy of the DataFrame with binary risk flag columns added.
    """
    df = df.copy()

    df["high_temperature"] = (df["temperature"] > 85).astype(int)
    df["low_temperature"] = (df["temperature"] < 65).astype(int)
    df["high_pressure"] = (df["pressure"] > 37).astype(int)
    df["low_pressure"] = (df["pressure"] < 23).astype(int)
    df["long_process_time"] = (df["process_time"] > 85).astype(int)

    return df

def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical manufacturing variables using one-hot encoding.

    Equipment ID and recipe are nominal categorical features, meaning they do
    not have natural order. One-Hot encoding converts them into binary
    indicator columns that can be used by machine learning models. 

    Args:
        df: Input DataFrame containing categorical columns.

    Returns:
        A copy of the DataFrame with categorical columns one-hot encoded. 
    """
    df = df.copy()

    df = pd.get_dummies(
        df,
        columns=["equipment_id", "recipe"],
        dtype=int
    )
    return df

def save_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Save the featured dataset to a CSV file. 

    Args: 
        df: DataFrame to save. 
        output_path: Path where the featured dataset should be saved.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    input_file = "data/processed/cleaned_synthetic_manufacturing_data.csv"
    output_path = "data/featured/featured_synthetic_manufacturing_data.csv"

    df = load_data(input_file)
    df = create_interaction_features(df)
    df = create_ratio_features(df)
    df = create_risk_flags(df)
    df = encode_categorical_features(df)

    save_data(df, output_path)

if __name__ == "__main__":
    main()