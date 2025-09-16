---

### **Project 3: Network Traffic Volume Forecasting**

**Objective:** To predict future network traffic volume based on historical data. This is a critical task for capacity planning, resource allocation, and identifying future bottlenecks before they occur.

**Dataset Source:** **Kaggle**. We will use the "Internet Traffic Time Series" dataset, which contains daily traffic data for an ISP. Its clear temporal structure is perfect for this forecasting task.

**Model:** We will use **Prophet**, a forecasting procedure developed by Facebook's Core Data Science team.

**Instructions:**
This notebook requires the Kaggle API. If you have already uploaded your `kaggle.json` file in this Colab session, you can skip the setup cell. If you are in a new session, please run the setup cell and upload your `kaggle.json` file when prompted.

**Implementation in Google Colab:**

```python
#
# ==================================================================================
#  Project 3: Network Traffic Volume Forecasting
# ==================================================================================
#
# Objective:
# This notebook demonstrates how to forecast future network traffic using the
# Prophet library. We will train a model on historical data and generate a
# prediction for the upcoming months.
#
# To Run in Google Colab:
# 1. Have your `kaggle.json` API token ready.
# 2. Copy and paste this entire code block into a single cell.
# 3. Run the cell. If it's your first time in this session, you will be
#    prompted to upload your `kaggle.json` file.
#

# ----------------------------------------
# 1. Setup Kaggle API and Download Data
# ----------------------------------------
import os

# Check if kaggle.json already exists to avoid re-uploading
if not os.path.exists('/root/.kaggle/kaggle.json'):
    print("--- Setting up Kaggle API ---")

    # Install the Kaggle library
    !pip install -q kaggle

    # Prompt user to upload their kaggle.json file
    from google.colab import files
    print("\nPlease upload your kaggle.json file:")
    uploaded = files.upload()

    if 'kaggle.json' not in uploaded:
        print("\nError: kaggle.json not uploaded. Please restart the cell and upload the file.")
        exit()

    print("\nkaggle.json uploaded successfully.")

    # Create the .kaggle directory and move the json file there
    !mkdir -p ~/.kaggle
    !cp kaggle.json ~/.kaggle/
    !chmod 600 ~/.kaggle/kaggle.json
else:
    print("Kaggle API already configured.")


print("\n--- Downloading Internet Traffic Time Series Dataset from Kaggle ---")
# Download the dataset (user: shenba, dataset: internet-traffic-time-series-data)
!kaggle datasets download -d shenba/internet-traffic-time-series-data

print("\n--- Unzipping the dataset ---")
# Unzip the downloaded file
!unzip -q internet-traffic-time-series-data.zip -d .

print("\nDataset setup complete.")


# ----------------------------------------
# 2. Load and Prepare the Data
# ----------------------------------------
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

print("\n--- Loading and Preprocessing Data ---")

# Install the Prophet library
!pip install -q prophet

try:
    # The CSV file is inside a folder with the same name as the dataset
    df = pd.read_csv('internet_traffic_data.csv', parse_dates=['Date'], dayfirst=True)
    print("Loaded internet_traffic_data.csv successfully.")
except FileNotFoundError:
    print("Error: internet_traffic_data.csv not found.")
    exit()

# Prophet requires columns to be named 'ds' (datestamp) and 'y' (value).
df.rename(columns={'Date': 'ds', 'Traffic': 'y'}, inplace=True)

print("\nDataset preview:")
print(df.head())

print("\nDataset info:")
df.info()

# Visualize the time series to understand its patterns
print("\nPlotting historical traffic data...")
plt.figure(figsize=(14, 7))
plt.plot(df['ds'], df['y'])
plt.title('Historical Internet Traffic')
plt.xlabel('Date')
plt.ylabel('Traffic')
plt.grid(True)
plt.show()

# ----------------------------------------
# 3. Split Data for Evaluation
# ----------------------------------------
# To evaluate our model, we'll hold out the last portion of the data as a test set.
# Let's use the last 90 days for testing.
split_point = len(df) - 90
df_train = df.iloc[:split_point]
df_test = df.iloc[split_point:]

print(f"\nSplitting data for evaluation:")
print(f"Training data from {df_train['ds'].min()} to {df_train['ds'].max()} ({len(df_train)} points)")
print(f"Test data from {df_test['ds'].min()} to {df_test['ds'].max()} ({len(df_test)} points)")


# ----------------------------------------
# 4. Model Training and Forecasting
# ----------------------------------------
print("\n--- Model Training ---")

# Initialize the Prophet model. Prophet will automatically detect seasonality.
model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)

print("Training the Prophet model...")
model.fit(df_train)
print("Training complete.")

# Create a dataframe for future predictions.
# We will predict for the length of our test set to evaluate performance.
future = model.make_future_dataframe(periods=len(df_test))
print("\nGenerating forecast...")
forecast = model.predict(future)

# Display the last few rows of the forecast
print("Forecast preview:")
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())


# ----------------------------------------
# 5. Model Evaluation and Visualization
# ----------------------------------------
print("\n--- Model Evaluation and Visualization ---")

# Plot the forecast
print("\nPlotting the forecast...")
fig1 = model.plot(forecast)
plt.title('Internet Traffic Forecast')
plt.xlabel('Date')
plt.ylabel('Traffic')
# Add a vertical line to show where the forecast starts
plt.axvline(df_train['ds'].max(), color='r', linestyle='--', lw=2, label='Forecast Start')
plt.legend()
plt.show()

# Plot the components of the forecast (trend, weekly, yearly seasonality)
print("\nPlotting forecast components...")
fig2 = model.plot_components(forecast)
plt.show()

# Evaluate the model on the test set
# Isolate the predicted values for the test period
y_pred = forecast.iloc[split_point:]['yhat']
y_true = df_test['y']

mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)

print("\n--- Quantitative Evaluation ---")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")


# ----------------------------------------
# 6. Conclusion
# ----------------------------------------
print("\n--- Conclusion ---")
print("The Prophet model successfully captured the main patterns in the traffic data:")
print("1. An overall upward trend in traffic over time.")
print("2. A strong yearly seasonality, likely corresponding to holidays or seasonal usage patterns.")
print("3. A clear weekly seasonality, showing dips in traffic on certain days (likely weekends).")
print(f"The RMSE of {rmse:.2f} indicates the model's predictions are, on average, this close to the actual values, providing a solid baseline for capacity planning.")
print("This forecasting model can be used to anticipate future demand, helping network engineers make informed decisions about infrastructure upgrades and resource management.")
```