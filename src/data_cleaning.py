"""
Data Cleaning Module

This module performs data cleaning for the synthetic manufacturing dataset. 

Main tasks include:
- Loading raw manufacturing data
- Checking for duplicate rows 
- Converting columns to appropriate data types
- Replacing invalid values with NaN
- Removing rows with missing values
- Saving the cleaned dataset

The cleaned dataset is stored in the processed data directory
and will be used for feature engineering and model training. 
"""
from pathlib import Path
import pandas as pd

def load_data(input_file: str) -> pd.DataFrame:
    """
    Load manufacturing dataset from a CSV file.

    Args: 
        input_file: Path to the input CSV file.

    Returns:
        Loaded pandas DataFrame.    
    """

    df = pd.read_csv(input_file)
    return df

def check_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check for duplicate rows in the dataset. 

    This function counts duplicate rows and prints 
    the total number of duplicate rows.

    Args:
        df: Input pandas DataFrame.

    Returns:
        Original pandas DataFrame.
    """

    duplicate_count = df.duplicated().sum()
    print(f"Total number of duplicate rows: {duplicate_count}")
    return df


def convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert selected columns to numeric data types.

    Any values that cannot be converted to numeric 
    are replaced with NaN using 'errors="coerce"'. 
    These NaN values will be handled later in the data cleaning process. 
    
    Args:
        df: input pandas DataFrame.

    Returns:
        DataFrame with converted numeric columns. 
    """

    df = df.copy()

    numeric_columns = ["temperature", "pressure", "process_time"]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce") 

    return df
    
def fix_invalid_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect and replace invalid values with NaN.

    Values outside predefined valid ranges are considered 
    invalid and replaced with NaN. 

    Args:
        df: Input pandas DataFrame.

    Returns:
        Cleaned DataFrame with invalid values replaced by NaN. 
    """

    df = df.copy()

    valid_ranges = {
        "temperature": (40, 120),
        "pressure": (0, 100),
        "process_time": (1, 300)
    }

    for column, (min_value, max_value) in valid_ranges.items(): 
        invalid_mask = (
            (df[column] < min_value) |
            (df[column] > max_value)
        )  

        invalid_count = invalid_mask.sum()
        print(f"{column}: {invalid_count} invalid values" )
        df.loc[invalid_mask, column] = pd.NA

    return df

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows containing missing values.

    This function removes all rows with missing values (NaN),
    prints the number of rows before and after cleaning,
    and reports how many rows were removed.

    Args:
        df: Input pandas DataFrame.

    Returns:
        Cleaned pandas DataFrame with missing values removed. 
    """

    df = df.copy()
    before_rows = len(df)
    df = df.dropna()
    after_rows = len(df)

    print(f"Rows before cleaning: {before_rows}")
    print(f"Rows after cleaning : {after_rows}")
    print(f"Rows removed: {before_rows - after_rows}")

    return df

def save_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Save the cleaned dataset to a CSV file. 

    If the output directory does not exist,
    it will be created automatically. 

    Args:
        df: DataFrame to save.
        output_path: Destination CSV file path.

    Returns:
        None    
    """

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def cleaned_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the complete data cleaning pipeline.
    """

    df = check_duplicates(df)
    df = convert_data_types(df)
    df = fix_invalid_values(df)
    df = handle_missing_values(df)

    return df

def main() -> None:
    input_file = "data/raw/synthetic_manufacturing_data.csv"
    output_path = "data/processed/cleaned_synthetic_manufacturing_data.csv"

    df = load_data(input_file)
    df = cleaned_data(df)
    save_data(df, output_path)
    
if __name__ == "__main__":
    main()