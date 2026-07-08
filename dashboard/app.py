"""
Streamlit dashboard for manufacturing defect prediction.
"""
import sys
from pathlib import Path

import plotly.express as px
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.predict import load_model
from src.predict import make_prediction

def highlight_defect(row):
    """
    Highlight rows containing predicted defects.
    """
    if row["Prediction"] == "❌ Defect":
        return["background-color: #ffdddd"] * len(row)
    return [""]*len(row)

def create_display_dataframe(result_df: pd.DataFrame) -> pd.DataFrame:
    """
    Display formatted DataFrame for dashboard display.
    """
    display_df = result_df.copy()
    
    display_df["predicted_defect"] = display_df["predicted_defect"].map({
        0: "✅ Normal",
        1: "❌ Defect"
    })    

    display_df = display_df[
        [
            "temperature",
            "pressure",
            "predicted_defect"
        ]
    ]
    display_df = display_df.rename(
        columns={
            'temperature': "Temperature",
            'pressure': "Pressure",
            'predicted_defect': "Prediction"
        }
    )               

    return display_df

def show_prediction_results(display_df: pd.DataFrame) -> None:
    """
    Display formatted prediction results in the dashboard.
    """
    style_df = display_df.style.apply(
        highlight_defect,
        axis=1
    )

    style_df = st.dataframe(style_df, hide_index=True)   


def show_metrics(result_df: pd.DataFrame) -> pd.Series:
    """
    Display prediction summary metrics and return prediction counts. 
    """
    defect_counts = result_df["predicted_defect"].map({
        0: "Normal",
        1: "Defect"
    }).value_counts()

    total_samples = len(result_df)
    normal_count = defect_counts.get("Normal", 0)
    defect_count = defect_counts.get("Defect", 0)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "📦  Total Samples",
            total_samples
        )

    with col2:
        st.metric(
            "✅ Normal",
            normal_count
        )

    with col3:
        st.metric(
            "❌ Defect",
            defect_count
        )
    return defect_counts

def show_chart(defect_counts):
    """
    Display bar and pie charts for prediction results. 
    """
    col1, col2 = st.columns(2)

    with col1:
        bar_df = defect_counts.reset_index()
        bar_df.columns = ["Prediction", "Count"]

        fig = px.bar(
            bar_df,
            x="Prediction",
            y="Count",
            title="Defect Prediction Count"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        pie_df = defect_counts.reset_index()
        pie_df.columns = ["Prediction", "Count"]

        fig = px.pie(
            pie_df,
            names="Prediction",
            values="Count",
            title="Prediction Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)


def download_results(result_df: pd.DataFrame) -> None:
    """
    Provide a download button for prediction results.
    """
    csv = result_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Prediction Results",
        data=csv,
        file_name="prediction_result.csv",
        mime="text/cvs"
    )

def main() -> None:
    """
    Run the Streamlit dashboard fir manufacturing defect prediction. 
    """
    st.set_page_config(
        page_title="Manufacturing Defect Detection",
        page_icon="🏭",
        layout="wide"
    )

    st.title("🏭 Manufacturing Defect Detection Dashboard")
    st.write("Upload a manufacturing dataset to analyze and predict product defects using the trained Gradient Boosting model.")

    upload_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"]
    )

    if upload_file is not None:
        df = pd.read_csv(upload_file)

        st.subheader("Uploaded Data")
        st.dataframe(df.head())

        model = load_model("models/gradient_boosting_model.pkl")

        if st.button("Run prediction"):
            result_df = make_prediction(model, df)

            display_df = create_display_dataframe(result_df)

            st.subheader("Prediction Results")

            show_prediction_results(display_df)
                    
            st.subheader("Prediction Overview")

            defect_counts = show_metrics(result_df)

            show_chart(defect_counts)

            download_results(result_df)


if __name__=="__main__":
    main()
