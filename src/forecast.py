import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression

# ----------------------------
# Load data
# ----------------------------
train_df = pd.read_csv("./data/train.csv")
stores = pd.read_csv("./data/stores.csv")

# Merge
data = train_df.merge(stores, on="Store", how="left")

# ----------------------------
# Data cleaning
# ----------------------------
data["Date"] = pd.to_datetime(data["Date"])
data = data.sort_values(by="Date")
data = data.dropna()

# ----------------------------
# Time series aggregation
# ----------------------------
ts = data.groupby("Date")["Weekly_Sales"].sum().reset_index()
ts = ts.sort_values("Date")

# ----------------------------
# Baseline model
# ----------------------------
split_date = "2012-01-01"

train_ts = ts[ts["Date"] < split_date].copy()
test_ts = ts[ts["Date"] >= split_date].copy()

train_ts["MA_4"] = train_ts["Weekly_Sales"].rolling(window=4).mean()
last_ma = train_ts["MA_4"].iloc[-1]

test_ts["Baseline"] = last_ma

mae_baseline = mean_absolute_error(test_ts["Weekly_Sales"], test_ts["Baseline"])

# ----------------------------
# Feature engineering
# ----------------------------
ts["Year"] = ts["Date"].dt.year
ts["Month"] = ts["Date"].dt.month
ts["Week"] = ts["Date"].dt.isocalendar().week.astype(int)

train_ml = ts[ts["Date"] < split_date]
test_ml = ts[ts["Date"] >= split_date]

features = ["Year", "Month", "Week"]

X_train = train_ml[features]
y_train = train_ml["Weekly_Sales"]

X_test = test_ml[features]
y_test = test_ml["Weekly_Sales"]

# ----------------------------
# ML Model
# ----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

preds = model.predict(X_test)

# ----------------------------
# Results
# ----------------------------
test_ml = test_ml.copy()
test_ml["ML"] = preds
test_ml["Baseline"] = last_ma

# Risk classification
test_ml["Risk"] = test_ml.apply(
    lambda x: "Overestimate" if x["ML"] > x["Weekly_Sales"] else "Underestimate",
    axis=1
)

# Errors
test_ml["Error_ML"] = abs(test_ml["Weekly_Sales"] - test_ml["ML"])
test_ml["Error_Baseline"] = abs(test_ml["Weekly_Sales"] - test_ml["Baseline"])

mae_ml = mean_absolute_error(y_test, preds)

# ----------------------------
# Print results
# ----------------------------
print(f"Baseline MAE: {mae_baseline:,.2f}")
print(f"ML MAE: {mae_ml:,.2f}")

# ----------------------------
# Visualization
# ----------------------------
plt.figure(figsize=(12, 5))
plt.plot(test_ml["Date"], y_test, label="Actual Demand")
plt.plot(test_ml["Date"], test_ml["Baseline"], label="Baseline Forecast")
plt.plot(test_ml["Date"], test_ml["ML"], label="ML Forecast")

plt.legend()
plt.title("Demand vs Forecast")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.show()

# ----------------------------
# Export
# ----------------------------
test_ml.to_csv("./outputs/forecast_results.csv", index=False)