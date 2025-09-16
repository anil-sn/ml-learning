---

### **Project 31: Predicting Cloud Network Egress Costs**

**Objective:** To build a time-series forecasting model that can predict the daily network egress costs for a cloud environment based on historical usage patterns.

**Dataset Source:** **Synthetically Generated**. We will create a realistic time-series dataset of daily egress data. The data will include common real-world patterns like weekly seasonality (lower traffic on weekends), overall growth trends, and random spikes.

**Model:** We will use **Prophet**, a powerful and easy-to-use time-series forecasting library from Facebook. It is specifically designed to handle time-series data with multi-seasonality (like weekly and yearly patterns) and trends, making it a perfect fit for this problem.

**Instructions:**
This notebook is fully self-contained and does not require any external files or APIs. Simply run the entire code block in Google Colab.

**Implementation in Google Colab:**
```python
#
# ==================================================================================
#  Project 31: Predicting Cloud Network Egress Costs
# ==================================================================================
#
# Objective:
# This notebook builds a time-series model using Prophet to forecast daily
# cloud egress bandwidth and its associated costs.
#
# To Run in Google Colab:
# Copy and paste this entire code block into a single cell and run it.
#

# ----------------------------------------
# 1. Import Necessary Libraries
# ----------------------------------------
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# ----------------------------------------
# 2. Synthetic Cloud Egress Data Generation
# ----------------------------------------
print("--- Generating Synthetic Daily Cloud Egress Dataset ---")

# Simulation parameters
days = 730 # 2 years of data
cost_per_gb = 0.05 # A typical cloud provider egress cost
start_date = pd.to_datetime('2022-01-01')

# Create a date range
dates = pd.date_range(start_date, periods=days, freq='D')

# --- Create realistic patterns ---
# 1. Overall growth trend
trend = np.linspace(500, 1500, days) # Start at 500 GB/day, grow to 1500 GB/day
# 2. Weekly seasonality (lower usage on weekends)
weekday = dates.dayofweek
weekly_seasonality = np.sin(weekday * (2 * np.pi / 7)) * 100 + 50
weekly_seasonality[weekday >= 5] *= 0.2 # Reduce weekend traffic by 80%
# 3. Random noise and spikes
noise = np.random.normal(0, 50, days)
spikes = np.random.choice([0, 1], size=days, p=[0.97, 0.03]) * np.random.uniform(500, 1000, days)

# Combine patterns to get total egress GB
egress_gb = trend + weekly_seasonality + noise + spikes
egress_gb = np.maximum(100, egress_gb) # Ensure a minimum egress

# Calculate cost
cost_usd = egress_gb * cost_per_gb

# Create the DataFrame
df = pd.DataFrame({'ds': dates, 'y': cost_usd})
print(f"Dataset generation complete. Created {len(df)} daily records.")
print("\nDataset Sample:")
print(df.sample(5))

# Visualize the generated data
print("\nVisualizing the historical data...")
df.plot(x='ds', y='y', figsize=(14, 7), title='Historical Daily Egress Cost')
plt.ylabel('Cost (USD)')
plt.xlabel('Date')
plt.grid(True)
plt.show()


# ----------------------------------------
# 3. Data Splitting (Time-Series)
# ----------------------------------------
print("\n--- Splitting Data for Training and Testing ---")

# For time-series, we train on the past and test on the future.
split_point = days - 90 # Hold out the last 90 days for testing
df_train = df.iloc[:split_point]
df_test = df.iloc[split_point:]
print(f"Training data: {len(df_train)} days")
print(f"Test data: {len(df_test)} days")


# ----------------------------------------
# 4. Model Training with Prophet
# ----------------------------------------
print("\n--- Model Training ---")

# Prophet is powerful because it automatically detects trends and seasonality.
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False # Our data is daily, so this is not needed
)

print("Training the Prophet model...")
model.fit(df_train)
print("Training complete.")


# ----------------------------------------
# 5. Forecasting and Evaluation
# ----------------------------------------
print("\n--- Forecasting Future Costs ---")

# Create a future dataframe for the next 90 days (the length of our test set)
future = model.make_future_dataframe(periods=90)
forecast = model.predict(future)

print("Forecast generated. Sample of forecast data:")
# `yhat` is the predicted value, `yhat_lower` and `yhat_upper` are the uncertainty interval.
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

# --- Plot the forecast ---
print("\nVisualizing the forecast...")
fig1 = model.plot(forecast)
# Add a vertical line to show where the forecast begins
plt.axvline(df_train['ds'].max(), color='r', linestyle='--', lw=2, label='Forecast Start')
plt.title('Cloud Egress Cost Forecast')
plt.xlabel('Date')
plt.ylabel('Cost (USD)')
plt.legend()
plt.show()

# --- Plot the forecast components ---
# This shows us the individual patterns Prophet has learned.
print("\nVisualizing the learned components...")
fig2 = model.plot_components(forecast)
plt.show()


# ----------------------------------------
# 6. Quantitative Evaluation
# ----------------------------------------
print("\n--- Quantitative Model Evaluation ---")
# Compare the predicted values (`yhat`) with the actual values (`y`) in the test set
y_true = df_test['y'].values
y_pred = forecast.iloc[split_point:]['yhat'].values

mae = mean_absolute_error(y_true, y_pred)
# Calculate Mean Absolute Percentage Error (MAPE)
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print(f"Mean Absolute Error (MAE) on test set: ${mae:.2f}")
print(f"  (On average, the forecast for a given day is off by ${mae:.2f})")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"  (On average, the forecast is {mape:.2f}% different from the actual cost)")


# ----------------------------------------
# 7. Conclusion
# ----------------------------------------
print("\n--- Conclusion ---")
print("The Prophet model successfully learned the complex patterns in the historical egress data and generated an accurate forecast.")
print("Key Takeaways:")
print("- The model's component plot clearly shows that it correctly identified the strong weekly seasonality (low costs on weekends) and the overall positive growth trend, which are the main drivers of cost.")
print(f"- With a low MAPE of {mape:.2f}%, the model provides a reliable financial planning tool. A finance department could use this forecast for accurate cloud budget allocation.")
print("- From a network engineering perspective, this model serves as a powerful anomaly detection system. If the *actual* egress cost on a given day is significantly higher than the `yhat_upper` value of the forecast, it's a strong indicator that something unexpected has happened.")
print("- This could trigger an alert for the engineering team to investigate, allowing them to discover a misconfigured application, a data exfiltration attack, or a new service that was deployed without proper cost controls, preventing massive, unforeseen cloud bills.")
```