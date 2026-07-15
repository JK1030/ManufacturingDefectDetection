"""
Streamlit dashboard for manufacturing defect prediction.
"""

import pandas as pd
import plotly.express as px
import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000/predict"

REQUIRED_COLUMNS = [
    "lot_id",
    "equipment_id",
    "recipe",
    "temperature",
    "pressure",
    "process_time",
]


def highlight_defect(row: pd.Series) -> list[str]:
    """
    Highlight rows containing predicted defects.
    """
    if row["Prediction"] == "❌ Defect":
        return ["background-color: #ffdddd"] * len(row)

    return [""] * len(row)


def validate_uploaded_file(df: pd.DataFrame) -> bool:
    """
    Validate the uploaded raw manufacturing CSV.
    """
    if df.empty:
        st.error("Uploaded CSV file is empty.")
        return False

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        st.error(
            "Uploaded CSV is missing required columns: "
            f"{missing_columns}"
        )
        return False

    return True


def request_predictions(
    df: pd.DataFrame,
    filename: str,
) -> tuple[pd.DataFrame, dict]:
    """
    Send the raw manufacturing DataFrame to FastAPI as a CSV file.
    """
    api_input_df = df.drop(
        columns=["defect"],
        errors="ignore",
    )

    csv_bytes = api_input_df.to_csv(
        index=False,
    ).encode("utf-8")

    files = {
        "file": (
            filename,
            csv_bytes,
            "text/csv",
        )
    }

    response = requests.post(
        API_URL,
        files=files,
        timeout=60,
    )

    if not response.ok:
        try:
            detail = response.json().get(
                "detail",
                response.text,
            )
        except ValueError:
            detail = response.text

        raise requests.exceptions.HTTPError(
            detail,
            response=response,
        )

    response_data = response.json()

    result_df = pd.DataFrame(
        response_data["predictions"]
    )

    return result_df, response_data


def create_display_dataframe(
    result_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a formatted DataFrame for dashboard display.
    """
    display_df = result_df.copy()

    display_df["predicted_defect"] = (
        display_df["predicted_defect"].map(
            {
                0: "✅ Normal",
                1: "❌ Defect",
            }
        )
    )

    display_df = display_df[
        [
            "lot_id",
            "temperature",
            "pressure",
            "process_time",
            "predicted_defect",
        ]
    ]

    display_df = display_df.rename(
        columns={
            "lot_id": "Lot ID",
            "temperature": "Temperature",
            "pressure": "Pressure",
            "process_time": "Process Time",
            "predicted_defect": "Prediction",
        }
    )

    return display_df


def show_prediction_results(
    display_df: pd.DataFrame,
) -> None:
    """
    Display formatted prediction results.
    """
    styled_df = display_df.style.apply(
        highlight_defect,
        axis=1,
    )

    st.dataframe(
        styled_df,
        hide_index=True,
        use_container_width=True,
    )


def show_metrics(
    result_df: pd.DataFrame,
) -> pd.Series:
    """
    Display prediction metrics and return prediction counts.
    """
    defect_counts = (
        result_df["predicted_defect"]
        .map(
            {
                0: "Normal",
                1: "Defect",
            }
        )
        .value_counts()
    )

    total_samples = len(result_df)
    normal_count = int(defect_counts.get("Normal", 0))
    defect_count = int(defect_counts.get("Defect", 0))

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("📦 Total Samples", total_samples)

    with col2:
        st.metric("✅ Normal", normal_count)

    with col3:
        st.metric("❌ Defect", defect_count)

    return defect_counts


def show_chart(
    defect_counts: pd.Series,
) -> None:
    """
    Display bar and pie charts.
    """
    chart_df = defect_counts.reset_index()
    chart_df.columns = ["Prediction", "Count"]

    col1, col2 = st.columns(2)

    with col1:
        bar_figure = px.bar(
            chart_df,
            x="Prediction",
            y="Count",
            title="Defect Prediction Count",
        )

        st.plotly_chart(
            bar_figure,
            use_container_width=True,
        )

    with col2:
        pie_figure = px.pie(
            chart_df,
            names="Prediction",
            values="Count",
            title="Prediction Distribution",
        )

        st.plotly_chart(
            pie_figure,
            use_container_width=True,
        )


def download_results(
    result_df: pd.DataFrame,
) -> None:
    """
    Provide a CSV download button.
    """
    csv_data = result_df.to_csv(
        index=False,
    ).encode("utf-8")

    st.download_button(
        label="Download Prediction Results",
        data=csv_data,
        file_name="prediction_result.csv",
        mime="text/csv",
    )


def main() -> None:
    """
    Run the Streamlit manufacturing dashboard.
    """
    st.set_page_config(
        page_title="Manufacturing Defect Detection",
        page_icon="🏭",
        layout="wide",
    )

    st.title(
        "🏭 Manufacturing Defect Detection Dashboard"
    )

    st.write(
        "Upload a raw manufacturing dataset to predict "
        "product defects using the trained model."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"],
    )

    st.write("uploaded_file=", uploaded_file)
    st.write("type =", type(uploaded_file))
    st.write("name =", uploaded_file.name)
    st.write("size =", uploaded_file.size)

    if uploaded_file is None:
        return

    try:
        raw_df = pd.read_csv(uploaded_file)

    except (
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        UnicodeDecodeError,
    ) as error:
        st.error(
            f"Could not read the uploaded CSV file: {error}"
        )
        return

    if not validate_uploaded_file(raw_df):
        return

    st.subheader("Uploaded Data")

    st.dataframe(
        raw_df.head(),
        hide_index=True,
        use_container_width=True,
    )

    if st.button("Run prediction"):
        try:
            with st.spinner(
                "Sending raw data to the prediction API..."
            ):
                result_df, response_data = request_predictions(
                    raw_df,
                    uploaded_file.name,
                )

            removed_rows = int(
                response_data.get("removed_rows", 0)
            )

            if removed_rows > 0:
                st.warning(
                    f"{removed_rows} rows were removed during "
                    "data cleaning because they contained "
                    "missing or invalid values."
                )

            if result_df.empty:
                st.warning(
                    "The API did not return any valid predictions."
                )
                return

            display_df = create_display_dataframe(
                result_df
            )

            st.subheader("Prediction Results")
            show_prediction_results(display_df)

            st.subheader("Prediction Overview")
            defect_counts = show_metrics(result_df)

            show_chart(defect_counts)
            download_results(result_df)

        except requests.exceptions.ConnectionError:
            st.error(
                "Could not connect to FastAPI. "
                "Make sure the API server is running."
            )

        except requests.exceptions.Timeout:
            st.error(
                "The prediction API request timed out."
            )

        except requests.exceptions.HTTPError as error:
            st.error(
                f"Prediction API error: {error}"
            )

        except requests.exceptions.RequestException as error:
            st.error(
                f"Request failed: {error}"
            )

        except Exception as error:
            st.error(
                f"Prediction failed: {error}"
            )


if __name__ == "__main__":
    main()
