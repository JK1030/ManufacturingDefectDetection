"""
FastAPI application for manufacturing defect prediction.

The API receives a raw manufacturing CSV file, validates the input,
cleans the data, performs feature engineering, and returns predictions.
"""

import io
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile


BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.data_cleaning import cleaned_data
from src.feature_engineering import engineer_features


MODEL_PATH = BASE_DIR / "models" / "gradient_boosting_model.pkl"
FEATURE_COLUMNS_PATH = BASE_DIR / "models" / "feature_columns.pkl"

REQUIRED_COLUMNS = [
    "lot_id",
    "equipment_id",
    "recipe",
    "temperature",
    "pressure",
    "process_time",
]


app = FastAPI(
    title="Manufacturing Defect Detection API",
    description=(
        "Upload a raw manufacturing CSV file and receive "
        "defect predictions."
    ),
    version="1.0.0",
)


if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file was not found: {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)

if FEATURE_COLUMNS_PATH.exists():
    feature_columns = list(
        joblib.load(FEATURE_COLUMNS_PATH)
    )
elif hasattr(model, "feature_names_in_"):
    feature_columns = list(
        model.feature_names_in_
    )
else:
    feature_columns = None


def validate_dataframe(df: pd.DataFrame) -> None:
    """
    Validate the uploaded raw manufacturing DataFrame.
    """
    if df.empty:
        raise HTTPException(
            status_code=400,
            detail="Uploaded CSV file is empty.",
        )

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise HTTPException(
            status_code=400,
            detail=(
                "Uploaded CSV is missing required columns: "
                f"{missing_columns}"
            ),
        )


@app.get("/")
def root() -> dict[str, str]:
    """
    Return the current API status.
    """
    return {
        "message": (
            "Manufacturing Defect Detection API is running."
        )
    }


@app.post("/predict")
async def predict_defects(
    file: UploadFile = File(...),
) -> dict[str, Any]:
    """
    Predict manufacturing defects from an uploaded raw CSV file.
    """
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file was uploaded.",
        )

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported.",
        )

    try:
        file_content = await file.read()

        if not file_content:
            raise HTTPException(
                status_code=400,
                detail="Uploaded CSV file is empty.",
            )

        try:
            raw_df = pd.read_csv(
                io.BytesIO(file_content)
            )

        except (
            pd.errors.EmptyDataError,
            pd.errors.ParserError,
            UnicodeDecodeError,
        ) as error:
            raise HTTPException(
                status_code=400,
                detail=f"Unable to read CSV file: {error}",
            ) from error

        validate_dataframe(raw_df)

        raw_df = raw_df.drop(
            columns=["defect"],
            errors="ignore",
        )

        original_row_count = len(raw_df)

        cleaned_df = cleaned_data(raw_df)

        if cleaned_df.empty:
            raise HTTPException(
                status_code=400,
                detail=(
                    "No valid rows remain after data cleaning. "
                    "Check missing values and valid numeric ranges."
                ),
            )

        result_df = cleaned_df[
            REQUIRED_COLUMNS
        ].copy()

        engineered_df = engineer_features(
            cleaned_df.copy()
        )

        feature_df = engineered_df.drop(
            columns=["lot_id", "defect"],
            errors="ignore",
        )

        if feature_columns is not None:
            feature_df = feature_df.reindex(
                columns=feature_columns,
                fill_value=0,
            )

        expected_features = getattr(
            model,
            "n_features_in_",
            None,
        )

        if (
            expected_features is not None
            and feature_df.shape[1] != expected_features
        ):
            raise HTTPException(
                status_code=500,
                detail=(
                    "Model feature mismatch. "
                    f"The model expects {expected_features} features, "
                    f"but preprocessing created {feature_df.shape[1]}. "
                    "Save the training feature columns as "
                    "models/feature_columns.pkl."
                ),
            )

        predictions = model.predict(
            feature_df
        )

        result_df["predicted_defect"] = (
            predictions.astype(int)
        )

        result_df["result"] = (
            result_df["predicted_defect"].map(
                {
                    0: "Normal",
                    1: "Defect",
                }
            )
        )

        prediction_records = (
            result_df.where(
                pd.notnull(result_df),
                None,
            )
            .to_dict(orient="records")
        )

        normal_count = int(
            (
                result_df["predicted_defect"] == 0
            ).sum()
        )

        defect_count = int(
            (
                result_df["predicted_defect"] == 1
            ).sum()
        )

        return {
            "filename": file.filename,
            "original_rows": original_row_count,
            "processed_rows": len(result_df),
            "removed_rows": (
                original_row_count - len(result_df)
            ),
            "normal_count": normal_count,
            "defect_count": defect_count,
            "predictions": prediction_records,
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {error}",
        ) from error

    finally:
        await file.close()
