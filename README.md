# DeepCSAT — Ecommerce

Predict Customer Satisfaction (CSAT) scores for ecommerce customer-support interactions using a trained TensorFlow model, a notebook-aligned preprocessing pipeline, and a Streamlit app for interactive inference.

## About
DeepCSAT is an ML inference project focused on **CSAT score prediction (1–5)** from customer support ticket/order interaction data. It includes:
- a reusable preprocessing pipeline aligned with notebook training logic,
- artifact generation for production-style inference,
- batch prediction via CLI,
- and an interactive Streamlit UI for single and CSV-based predictions.

## Suggested GitHub Topics
`machine-learning`, `tensorflow`, `streamlit`, `customer-satisfaction`, `csat`, `ecommerce`, `classification`, `data-preprocessing`, `model-inference`, `python`

## Repository Structure
- `app.py` — Streamlit app for single and batch CSAT prediction
- `inference/preprocessing.py` — notebook-aligned preprocessing implementation
- `inference/build_artifacts.py` — build and validate preprocessing artifacts
- `inference/predict.py` — batch prediction script (CSV in, CSV out)
- `artifacts/` — saved preprocessing artifact and feature schema
- `csat_model.h5` — trained neural network model
- `predictions.csv` — sample/generated predictions file

## How It Works
1. Load the trained model (`csat_model.h5`).
2. Apply the fitted preprocessing artifact (`artifacts/preprocessor.joblib`).
3. Produce class probabilities and predicted CSAT score (1 to 5).
4. Export predictions for downstream use.

## Setup
Use Python 3.10+.

Install required dependencies:

```bash
pip install pandas numpy scikit-learn joblib tensorflow streamlit
```

## Usage
From repository root:

### 1) Build preprocessing artifacts
```bash
python inference/build_artifacts.py \
  --data-path eCommerce_Customer_support_data.csv \
  --model-path csat_model.h5 \
  --output-dir artifacts
```

### 2) Run batch inference
```bash
python inference/predict.py \
  --input-csv eCommerce_Customer_support_data.csv \
  --output-csv predictions.csv \
  --model-path csat_model.h5 \
  --preprocessor-path artifacts/preprocessor.joblib
```

### 3) Launch Streamlit app
```bash
streamlit run app.py
```

## Streamlit Features
- **Single Prediction** tab with form inputs
- **Batch CSV Prediction** tab with file upload
- Predicted class probabilities (`prob_csat_1` ... `prob_csat_5`)
- Downloadable prediction output CSV

## Input Expectations
The pipeline expects columns used by preprocessing, including identifiers, categorical fields, date/time fields, and numeric fields such as:
- `channel_name`, `category`, `Sub-category`, `Product_category`
- `order_date_time`, `Issue_reported at`, `issue_responded`, `Survey_response_Date`
- `Item_price`, `connected_handling_time`

If columns are missing, the preprocessing code applies safe defaults where possible.

## Notes
- Ensure model and artifact paths are correct before running inference.
- If `artifacts/preprocessor.joblib` is missing, run `inference/build_artifacts.py` first.
- The current repository does not include a pinned `requirements.txt`; consider adding one for reproducible environments.
