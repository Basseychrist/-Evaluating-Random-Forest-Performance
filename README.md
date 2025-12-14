# Evaluating Random Forest Performance

This repo contains the notebook **Evaluating-random-forest-v1.ipynb**, which walks through training and evaluating a Random Forest regressor on the California Housing dataset. Below is a quick guide to what each code section does and why it is needed.

## Notebook walkthrough
- **Package check/install:** Ensures required libraries (numpy, pandas, scikit-learn, matplotlib, scipy) are present so all cells run without manual setup.
- **Imports:** Pulls in data handling (numpy, pandas), visualization (matplotlib), dataset loader (fetch_california_housing), model/metrics (RandomForestRegressor, train_test_split, mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score), and skewness (scipy.stats.skew).
- **Load data:** Fetches the California Housing features/targets (`X`, `y`) for modeling.
- **Describe dataset:** Prints the dataset description to understand feature meanings and target definition.
- **Train/test split (Exercise 1):** Splits data with 20% held out to estimate out-of-sample performance.
- **EDA summary:** Builds a DataFrame of training data, adds target, and calls `describe()` to inspect ranges/quantiles.
- **Price distribution plot:** Histograms target values to observe skewness and clipping near $500k.
- **Model fit/predict:** Trains a default `RandomForestRegressor` (100 trees, fixed seed) and predicts on the test set.
- **Metrics (MAE, MSE, RMSE, R²):** Quantifies average error magnitude, squared error, its root (in dollars), and variance explained.
- **Actual vs. predicted scatter:** Visual check of fit quality and bias across target range.
- **Residuals histogram (Exercise 4):** Shows error distribution and reports mean/std to spot bias or heavy tails.
- **Residuals vs. actual (Exercise 5):** Plots residuals ordered by actual price to detect systematic under/over-prediction trends.
- **Feature importances (Exercise 7):** Bar chart of Random Forest feature importances to interpret key drivers (e.g., median income, location proxies).
- **Discussion prompts (Exercises 6 & 8):** Reflection on residual trends, skewness, clipping effects, and whether standardization is needed.

## How to use
1) Open the notebook in VS Code or Jupyter.
2) Run the package check/install cell once to ensure dependencies.
3) Execute cells top to bottom; fill in exercise cells as prompted.
