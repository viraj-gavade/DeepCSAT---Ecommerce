from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf


MODEL_PATH = Path("csat_model.h5")
PREPROCESSOR_PATH = Path("artifacts/preprocessor.joblib")
TRAIN_CSV_PATH = Path("eCommerce_Customer_support_data.csv")


@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file missing: {MODEL_PATH}")
    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            f"Preprocessor artifact missing: {PREPROCESSOR_PATH}. "
            "Run inference/build_artifacts.py first."
        )

    model = tf.keras.models.load_model(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor


@st.cache_data
def get_reference_options() -> dict:
    if not TRAIN_CSV_PATH.exists():
        return {}

    df = pd.read_csv(TRAIN_CSV_PATH)
    option_cols = [
        "channel_name",
        "category",
        "Sub-category",
        "Product_category",
        "Tenure Bucket",
        "Agent Shift",
    ]

    options = {}
    for col in option_cols:
        if col in df.columns:
            values = sorted(df[col].dropna().astype(str).unique().tolist())
            options[col] = values
    return options


def score_to_label(score: int) -> str:
    return f"Predicted CSAT Score: {score}/5"


def build_single_record_form(options: dict) -> pd.DataFrame:
    col1, col2 = st.columns(2)

    with col1:
        channel = st.selectbox(
            "channel_name",
            options.get("channel_name", ["Inbound", "Outcall"]),
        )
        category = st.selectbox(
            "category",
            options.get("category", ["Order Related"]),
        )
        sub_category = st.selectbox(
            "Sub-category",
            options.get("Sub-category", ["General Query"]),
        )
        product_category = st.selectbox(
            "Product_category",
            options.get("Product_category", ["Electronics"]),
        )
        tenure_bucket = st.selectbox(
            "Tenure Bucket",
            options.get("Tenure Bucket", ["31-60"]),
        )
        agent_shift = st.selectbox(
            "Agent Shift",
            options.get("Agent Shift", ["Morning"]),
        )

    with col2:
        item_price = st.number_input("Item_price", min_value=0.0, value=500.0, step=10.0)
        handling_time = st.number_input(
            "connected_handling_time", min_value=0.0, value=60.0, step=1.0
        )
        issue_reported = st.text_input("Issue_reported at", value="2023-08-01 10:00")
        issue_responded = st.text_input("issue_responded", value="2023-08-01 10:10")
        order_date_time = st.text_input("order_date_time", value="2023-08-01 09:55")
        survey_date = st.text_input("Survey_response_Date", value="2023-08-01")

    customer_remarks = st.text_input("Customer Remarks", value="Unknown")

    row = {
        "Unique id": "single-row",
        "Order_id": "single-order",
        "channel_name": channel,
        "category": category,
        "Sub-category": sub_category,
        "Customer Remarks": customer_remarks,
        "order_date_time": order_date_time,
        "Issue_reported at": issue_reported,
        "issue_responded": issue_responded,
        "Survey_response_Date": survey_date,
        "Customer_City": "Unknown",
        "Product_category": product_category,
        "Item_price": item_price,
        "connected_handling_time": handling_time,
        "Agent_name": "Unknown",
        "Supervisor": "Unknown",
        "Manager": "Unknown",
        "Tenure Bucket": tenure_bucket,
        "Agent Shift": agent_shift,
    }

    return pd.DataFrame([row])


def run_prediction(df: pd.DataFrame, model, preprocessor) -> pd.DataFrame:
    X = preprocessor.transform(df)
    probs = model.predict(X, verbose=0)

    predicted_class_index = np.argmax(probs, axis=1)
    predicted_csat_score = predicted_class_index + 1

    out = df.copy()
    out["predicted_csat_score"] = predicted_csat_score
    for class_idx in range(probs.shape[1]):
        score = class_idx + 1
        out[f"prob_csat_{score}"] = probs[:, class_idx]

    return out


def main() -> None:
    st.set_page_config(page_title="CSAT Inference App", page_icon="📈", layout="wide")
    st.title("CSAT Inference Pipeline")
    st.caption("Notebook-aligned preprocessing + saved ANN model inference")

    model, preprocessor = load_artifacts()
    options = get_reference_options()

    st.info(
        f"Loaded model input shape: {model.input_shape[-1]} features | "
        f"Preprocessor output: {len(preprocessor.final_feature_order_)} features"
    )

    with st.expander("Final model features"):
        st.write(preprocessor.final_feature_order_)

    tab_single, tab_batch = st.tabs(["Single Prediction", "Batch CSV Prediction"])

    with tab_single:
        st.subheader("Single Record")
        row_df = build_single_record_form(options)

        if st.button("Predict Single Record", type="primary"):
            try:
                pred_df = run_prediction(row_df, model, preprocessor)
                score = int(pred_df["predicted_csat_score"].iloc[0])
                st.success(score_to_label(score))

                proba_cols = [c for c in pred_df.columns if c.startswith("prob_csat_")]
                st.bar_chart(pred_df[proba_cols].T)
                st.dataframe(pred_df)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

    with tab_batch:
        st.subheader("Batch CSV")
        uploaded = st.file_uploader("Upload CSV", type=["csv"])

        if uploaded is not None:
            batch_df = pd.read_csv(uploaded)
            st.write(f"Rows uploaded: {len(batch_df)}")
            st.dataframe(batch_df.head())

            if st.button("Predict Batch", type="primary"):
                try:
                    pred_df = run_prediction(batch_df, model, preprocessor)
                    st.success("Batch prediction complete.")
                    st.dataframe(pred_df.head())

                    csv_bytes = pred_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download Predictions CSV",
                        data=csv_bytes,
                        file_name="predictions.csv",
                        mime="text/csv",
                    )
                except Exception as exc:
                    st.error(f"Batch prediction failed: {exc}")


if __name__ == "__main__":
    main()
