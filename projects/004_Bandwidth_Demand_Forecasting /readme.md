
### **Project 4: Bandwidth Demand Forecasting**

#### **1. Objective**
To build a time-series model that accurately forecasts future bandwidth demand on a network segment. This project will introduce the team to the fundamentals of time-series analysis, including handling timestamps, identifying seasonality, and using specialized forecasting libraries.

#### **2. Business Value**
Accurate demand forecasting is essential for efficient network management:
*   **Capacity Planning:** Justifies and guides network upgrades by predicting when and where future capacity will be needed.
*   **Resource Allocation:** Dynamically allocate resources in a virtualized network environment based on predicted demand.
*   **Congestion Prevention:** Proactively identify future peak usage periods to prevent service degradation and ensure a high quality of experience for customers.

#### **3. Core Libraries**
*   `pandas`: For robust time-series data manipulation.
*   `matplotlib` & `seaborn`: For visualizing the time-series data and forecast results.
*   `prophet`: A powerful and user-friendly forecasting library developed by Facebook, designed to handle time-series data with strong seasonal patterns.

#### **4. Dataset**
*   **Primary Dataset:** **Hourly Energy Consumption** ([Verified Link on Kaggle](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption))
*   **Why it's a suitable proxy:** While it measures energy, this dataset is an excellent proxy for network traffic because it shares the same statistical properties. It has:
    *   **Strong multi-level seasonality:** Clear daily (e.g., high usage during the day, low at night) and weekly (e.g., different patterns on weekends) cycles.
    *   **Long-term trends:** Overall usage may increase or decrease over the years.
    *   **Noise and outliers:** Realistic, imperfect data.
    The forecasting techniques learned on this dataset are directly and immediately applicable to bandwidth data.

#### **5. Detailed Step-by-Step Guide**

**Step 1: Setup the Environment**
1.  Create a new project folder and a Python virtual environment.
    ```bash
    mkdir demand-forecasting
    cd demand-forecasting
    python -m venv venv
    source venv/bin/activate
    ```
2.  Install the necessary libraries. Note that `prophet` can sometimes have complex dependencies, so installing it cleanly is important.
    ```bash
    pip install pandas matplotlib seaborn prophet jupyterlab
    ```
3.  Start a Jupyter Lab session.
    ```bash
    jupyter lab
    ```

**Step 2: Load and Prepare the Time-Series Data**
1.  Download the `AEP_hourly.csv` file from the Kaggle dataset.
2.  In your Jupyter Notebook, load the data and prepare it for time-series analysis. This preparation is critical.
    ```python
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv('AEP_hourly.csv')

    # 1. Convert the 'Datetime' column to a proper datetime object
    df['Datetime'] = pd.to_datetime(df['Datetime'])

    # 2. Set the datetime column as the index of the DataFrame
    df.set_index('Datetime', inplace=True)

    # 3. Visualize the data to see the patterns
    df['AEP_MW'].plot(figsize=(15, 6), title='Hourly Energy Consumption (MW)')
    plt.ylabel('Megawatts')
    plt.show()
    ```

**Step 3: Prepare Data for Prophet**
1.  The `prophet` library requires the data to be in a DataFrame with two specific column names: `ds` for the timestamp and `y` for the value to be forecasted.
    ```python
    # Reset the index to turn the 'Datetime' index back into a column
    df_prophet = df.reset_index()
    # Rename the columns to the required format
    df_prophet.rename(columns={'Datetime': 'ds', 'AEP_MW': 'y'}, inplace=True)
    
    print(df_prophet.head())
    ```

**Step 4: Train the Forecasting Model**
1.  Instantiate and fit the Prophet model to the prepared DataFrame. Prophet automatically detects trends, daily, weekly, and yearly seasonality.
    ```python
    from prophet import Prophet

    # Initialize the model
    model = Prophet()

    # Fit the model to your data
    model.fit(df_prophet)
    ```

**Step 5: Generate and Visualize the Forecast**
1.  **Create a future dataframe:** Tell Prophet how far into the future you want to forecast. Let's forecast for one year (365 days).
    ```python
    # Create a DataFrame for future predictions
    future = model.make_future_dataframe(periods=365)
    print(future.tail())
    ```
2.  **Make predictions:** Use the `predict` method on the future DataFrame.
    ```python
    forecast = model.predict(future)
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
    ```
    *   `yhat` is the forecasted value.
    *   `yhat_lower` and `yhat_upper` create an uncertainty interval.

3.  **Plot the forecast:** Prophet has a convenient built-in plotting function.
    ```python
    fig1 = model.plot(forecast)
    plt.title("Forecasted Energy Demand for the Next Year")
    plt.xlabel("Date")
    plt.ylabel("Megawatts")
    plt.show()
    ```

**Step 6: Analyze Forecast Components**
1.  One of Prophet's most powerful features is its ability to show you the components of the forecast. This helps you understand the underlying patterns it has learned.
    ```python
    fig2 = model.plot_components(forecast)
    plt.show()
    ```
    *   This will generate plots for the overall **trend**, **weekly seasonality** (e.g., lower usage on weekends), and **yearly seasonality** (e.g., higher usage in summer/winter). This is invaluable for interpreting the results.

#### **6. Success Criteria**
*   The team can correctly load, parse, and index time-series data using Pandas.
*   The model is successfully trained using `prophet`, and a forecast extending one year into the future is generated.
*   The team can produce and interpret the main forecast plot, showing the historical data, the forecasted values (`yhat`), and the uncertainty interval.
*   The team can generate and explain the **components plot**, describing the weekly and yearly seasonal patterns identified by the model.

#### **7. Next Steps & Extensions**
*   **Evaluation:** Split the data into a training set (e.g., all data before the last year) and a test set (the last year). Train the model on the training set and compare its forecast to the actual values in the test set using metrics like Mean Absolute Error (MAE).
*   **Add External Regressors:** Add information about public holidays, which often have a significant impact on demand patterns. Prophet has a built-in mechanism for this.
*   **Compare Models:** Compare Prophet's performance to a classical time-series model like SARIMA (Seasonal AutoRegressive Integrated Moving Average) from the `statsmodels` library.
