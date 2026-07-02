"""
Synthetic manufacturing data generator.

This script generates sample manufacturing process data for a defect detection machine learning project.
"""
from pathlib import Path
import numpy as np
import pandas as pd

def generate_synthetic_data(num_rows: int = 1000, random_seed: int = 42) -> pd.DataFrame:
    '''
    Generate synthetic manufacturing process data.

    Args:
        num_rows: Number of rows to generate.
        random_seed: Random seed for reproducible results. 

    Returns:
        A pandas DataFrame containing synthetic manufacturing data. 
    '''

    np.random.seed(random_seed)

    data = {
        "lot_id": [f"LOT_{i:05d}" for i in range(1, num_rows + 1)],
        "equipment_id": np.random.choice(["EQP_A", "EQP_B", "EQP_C"], size=num_rows),
        "recipe": np.random.choice(["RCP_1", "RCP_2","RCP_3"], size=num_rows),
        "temperature": np.random.normal(loc=75, scale=5, size=num_rows),
        "pressure": np.random.normal(loc=30, scale=3, size=num_rows),
        "process_time": np.random.normal(loc=120, scale=15, size=num_rows)
    }

    df = pd.DataFrame(data)

    # Simple defect rule:
    # Defects are more likely when temperature or pressure is outside normal range.
    df["defect"] = (
        (df["temperature"] > 82)
        | (df["temperature"] < 68)
        | (df["pressure"] > 35)
        | (df["pressure"] < 25)
    ).astype(int)

    return df


def inject_data_quality_issues(
        df: pd.DataFrame,
        missing_rate: float = 0.02,
        outlier_rate: float = 0.01,
        random_seed: int = 42
) -> pd.DataFrame:
    '''
    Inject missing values and outliers into the synthetic dataset. 

    Args:
        df: Clean synthetic manufacturing DataFrame. 
        missing_rate: Percentage of rows to receive missing values.
        outlier_rate: Percentage of rows to receive outlier values.
        random_seed: Random seed for reproducibility

    Returns:
        DataFrame with injected data quality issues.
    '''
    np.random.seed(random_seed)

    df = df.copy()
    num_rows = len(df)

    numeric_columns = ["temperature", "pressure", "process_time"]

    # Inject missing values
    for column in numeric_columns:
        missing_indices = np.random.choice(
            df.index,
            size=int(num_rows * missing_rate),
            replace=False
        )

        df.loc[missing_indices, column] = np.nan

    # Inject outliers
    outlier_indices = np.random.choice(
        df.index,
        size=int(num_rows * outlier_rate),
        replace=False
    )

    df.loc[outlier_indices, "temperature"] = np.random.choice([20, 150], size=len(outlier_indices))
    df.loc[outlier_indices, "pressure"] = np.random.choice([1, 100], size=len(outlier_indices))
    df.loc[outlier_indices, "process_time"] = np.random.choice([-10, 500], size=len(outlier_indices))

    return df

def save_data(df: pd.DataFrame, output_path: str) -> None:
    '''
    Save DataFrame to a CSV file.

    Args: 
        df: DataFrame to save.
        output_path: Target CSV file path.
    '''

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    output_path = "data/raw/synthetic_manufacturing_data.csv"

    df = generate_synthetic_data(num_rows=1000, random_seed=42)
    df = inject_data_quality_issues(df, missing_rate=0.02, outlier_rate=0.01, random_seed=42)

    save_data(df, output_path)

    print(f"Data saved to: {output_path}")
    print(f"Rows: {len(df)}")
    print(f"Missing values: \n{df.isnull().sum()}")
    print(f"Defect rate: {df['defect'].mean(): .2%}")


if __name__ == "__main__":
    main()

