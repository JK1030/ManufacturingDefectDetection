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
        "lot_id" : [f"LOT_{i:05d}" for i in range(1,num_rows + 1)],
        "equipment_id" : np.random.choice(["EQP_A", "EQP_B", "EQP_C"], size=num_rows),
        "recipe" : np.random.choice(["RCP_1", "RCP_2","RCP_3"], size=num_rows),
        "temperature" : np.random.normal(loc=75, scale=5, size=num_rows),
        "pressure" : np.random.normal(loc=30, scale=3, size=num_rows),
        "process_time" : np.random.normal(loc=120, scale=15, size=num_rows)
    }

    df = pd.DataFrame(data)

    #Simple defect rule:
    #Defects are more likely when temperature or pressure is outside normal range.
    df["defect"] = (
        (df["temperature"] > 82)
        | (df["temperature"] < 68)
        | (df["pressure"] > 35)
        | (df["pressure"] < 25)
    ).astype(int)

    return df

def save_data(df: pd.DataFrame, output_path: str) -> None:
    '''
    Save DataFrame to a CSV file.

    Arg: 
        df: DataFrame to save.
        output_path: Target CVS file path.
    '''

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def main() -> None:
    output_path = "data/raw/synthetic_manufacturing_data.csv"

    df = generate_synthetic_data(num_rows= 1000, random_seed=42)
    save_data(df, output_path)

    print(f"Data saved to: {output_path}")
    print(f"Rows: {len(df)}")
    print(f"Defect rate: {df['defect'].mean(): .2%}")


if __name__=="__main__":
    main()