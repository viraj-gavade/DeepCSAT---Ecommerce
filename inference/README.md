# CSAT Inference Pipeline

This folder contains production-style inference code aligned to the notebook logic and the saved ANN model (`csat_model.h5`).

## What Is Encoded

- Technique: One-Hot Encoding using `pd.get_dummies(..., drop_first=True)`
- Categorical source columns encoded:
  - `channel_name`
  - `category`
  - `Product_category`
  - `Tenure Bucket`
  - `Agent Shift`
  - Any remaining object columns in `X` after dropping high-cardinality/irrelevant fields

## Feature Engineering Used

1. Drop identifiers: `Unique id`, `Order_id`
2. Parse date columns:
   - `order_date_time`
   - `Issue_reported at`
   - `issue_responded`
   - `Survey_response_Date`
3. Create time-derived features:
   - `order_month`, `order_day`, `order_hour`
   - `response_time_minutes = (issue_responded - Issue_reported at)`
4. Missing value handling:
   - Numeric medians for `connected_handling_time`, `Item_price`, `response_time_minutes`, `order_month`, `order_day`, `order_hour`
   - Date mode for `order_date_time`, `Issue_reported at`, `issue_responded`
   - Categorical fill with `Unknown`
5. Text normalization: lower/strip `Customer Remarks`
6. Outlier capping (IQR clip, twice as done in notebook)
7. Drop time helpers before final modeling path: `order_month`, `order_day`, `order_hour`
8. Drop high-cardinality/irrelevant columns from modeling features
9. Correlation pruning (`> 0.85`)
10. Feature selection stack:
   - `VarianceThreshold(0.01)`
   - Correlation with target (`abs(corr) > 0.02`)
   - `SelectKBest(f_classif, k=min(20, n_features))`

Final ANN input shape expected by saved model: **15 features**.

## Files

- `preprocessing.py`: Reusable notebook-aligned preprocessing class
- `build_artifacts.py`: Fits and saves preprocessing artifacts
- `predict.py`: Batch CSV inference script

## Usage

From workspace root:

1. Build artifacts:

```bash
python inference/build_artifacts.py \
  --data-path eCommerce_Customer_support_data.csv \
  --model-path csat_model.h5 \
  --output-dir artifacts
```

2. Run batch prediction:

```bash
python inference/predict.py \
  --input-csv eCommerce_Customer_support_data.csv \
  --output-csv predictions.csv \
  --model-path csat_model.h5 \
  --preprocessor-path artifacts/preprocessor.joblib
```

3. Launch app:

```bash
streamlit run app.py
```
