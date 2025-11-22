# 🌟 uncertainty-retail-forecasting

### *End-to-End Retail Demand Forecasting with Temporal Fusion Transformers (TFT)*

**Accurate predictions • Uncertainty intervals • Deep learning • 500
SKU-level time series**

This project builds a **production-grade retail demand forecasting
system** using the **Temporal Fusion Transformer (TFT)** with
**probabilistic outputs**.\
It forecasts daily sales for **500 store--item combinations**, compares
against classical baselines, and quantifies uncertainty using quantile
regression.

🎯 **Goal:** Build an industry-level system that helps retailers
optimize inventory, reduce stockouts, and plan supply more effectively.

------------------------------------------------------------------------

# 🔥 Key Features

### ✔ Temporal Fusion Transformer (TFT)

-   Deep learning model designed for time series\
-   Captures long-term patterns + attention\
-   Learns complex interactions between stores/items\
-   Supports probabilistic forecasting

### ✔ Probabilistic Forecasting (Uncertainty)

Model outputs include: - **p10** -- lower confidence bound\
- **p50** -- median forecast\
- **p90** -- upper confidence bound

Used for: - Inventory buffers\
- Risk-aware planning\
- Safety stock decisions

### ✔ Benchmarking Against Classic Models

We compare TFT against: - **Naive** - **Moving Average (7-day)** -
**Seasonal Naive (weekly)**

### ✔ Evaluation Across 500 SKUs

For each store--item series: - RMSE\
- MAE\
- MAPE\
- Prediction intervals

------------------------------------------------------------------------

# 📊 Results Summary

### 🔹 Performance Comparison (500 SKUs)

  Model      Mean RMSE ↓   Median RMSE ↓
  ---------- ------------- ---------------
  Naive      11.73         10.97
  MA(7)      17.97         17.84
  Seasonal   20.55         20.40
  **TFT**    **7.95**      **7.59**

### 🏆 Key Insight

> **TFT improves accuracy by 32--60% compared to baseline models.**

It is the only model that consistently learns daily retail patterns
across all SKUs.

------------------------------------------------------------------------

# 📈 Example Forecast Visualization

A single SKU forecast showing uncertainty (p10--p90):

-   **Black:** Train\
-   **Gray:** Validation\
-   **Blue:** Median forecast\
-   **Green dashed:** p10\
-   **Red dashed:** p90\
-   **Shaded:** 80% prediction interval

------------------------------------------------------------------------

# 🧠 Modeling Pipeline

    Raw Data (store, item, date, sales)
            ↓
    Data Cleaning & EDA
            ↓
    Feature Engineering (lags, rolling stats)
            ↓
    Create 500 TimeSeries objects
            ↓
    Scale Each SKU Individually (Darts Scaler)
            ↓
    Train/Val Split per SKU
            ↓
    TFT Model (Quantile Regression)
            ↓
    Generate p10/p50/p90 predictions
            ↓
    Evaluate Across All SKUs (RMSE, MAE, MAPE)
            ↓
    Compare Against Baselines

------------------------------------------------------------------------

# 🛠 Tech Stack

  Component       Tools
  --------------- ----------------------------
  Language        Python 3.13
  Deep Learning   PyTorch, PyTorch Lightning
  Time Series     **Darts (u8darts)**
  Visualization   Matplotlib
  Data            Pandas, NumPy
  Notebooks       JupyterLab

------------------------------------------------------------------------

# 📁 Project Structure

    uncertainty-retail-forecasting/
    │
    ├── data/
    │   ├── raw/
    │   ├── processed/
    │
    ├── notebooks/
    │   ├── 01_eda.ipynb
    │   ├── 02_feature_engineering.ipynb
    │   ├── 03_baselines.ipynb
    │   ├── 04_tft_forecasting.ipynb
    │
    ├── README.md
    └── requirements.txt

------------------------------------------------------------------------

# 🚀 How to Run

### 1️⃣ Install dependencies

``` bash
pip install "u8darts[torch]"
pip install pandas numpy matplotlib scikit-learn
```

### 2️⃣ Launch Jupyter

``` bash
jupyter lab
```

### 3️⃣ Run notebooks in order:

    01 → 02 → 03 → 04

------------------------------------------------------------------------

# 📜 Insights & Takeaways

-   Daily retail demand is **noisy**\
-   Naive performs decently → strong short-term continuity\
-   MA(7) + Seasonal naive perform poorly\
-   **TFT reduces RMSE by 30--60%**\
-   Probabilistic forecasts provide inventory decision support
