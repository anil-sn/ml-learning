### **Project 6: Network Latency Prediction**

#### **1. Objective**
To build a regression model that predicts a continuous value (network latency in milliseconds) based on a set of network and path characteristics. This project will reinforce feature engineering skills and introduce XGBoost, a powerful gradient-boosting library that is a standard for high-performance modeling on tabular data. The key new skill in this project is **synthetic data generation**.

#### **2. Business Value**
Predicting latency enables proactive network management:
*   **Quality of Service (QoS) Monitoring:** Anticipate and mitigate performance degradation on critical paths before it impacts end-users.
*   **Optimized Routing:** In a Software-Defined Network (SDN), a latency prediction model can inform the controller to make smarter, real-time routing decisions.
*   **SLA Management:** Help ensure Service Level Agreements (SLAs) are met by flagging paths that are predicted to exceed latency thresholds.

#### **3. Core Libraries**
*   `pandas` & `numpy`: For creating and manipulating the dataset.
*   `scikit-learn`: For splitting data, and for training a baseline `LinearRegression` model.
*   `xgboost`: The primary, high-performance gradient boosting model.
*   `seaborn` & `matplotlib`: For visualization of correlations and results.

#### **4. Dataset**
*   **Approach:** **Synthetic Data Generation**.
*   **Why:** A perfect, public dataset containing all the desired features (e.g., path distance, device count, real-time traffic load, and measured latency) is not readily available. Therefore, we will create our own realistic, synthetic dataset. This is a crucial engineering skill that gives us full control over the data's characteristics and ensures the features are directly relevant to our domain.

#### **5. Detailed Step-by-Step Guide**

**Step 1: Setup the Environment**
1.  Create a project folder and a Python virtual environment.
    ```bash
    mkdir latency-prediction
    cd latency-prediction
    python -m venv venv
    source venv/bin/activate
    ```
2.  Install the necessary libraries.
    ```bash
    pip install pandas numpy scikit-learn xgboost seaborn matplotlib jupyterlab
    ```
3.  Start a Jupyter Lab session.
    ```bash
    jupyter lab
    ```

**Step 2: Generate the Synthetic Dataset**
This is the core creative step of the project. We will define the "rules" of our simulated network.
1.  In your Jupyter Notebook, create a script to generate a Pandas DataFrame.
2.  **Define the Features (Inputs):**
    *   `distance_km`: The physical distance of the fiber path. This should be a primary driver of latency.
    *   `traffic_load_gbps`: The current traffic volume on the path. Higher load should increase latency.
    *   `device_count`: The number of hops (routers, switches) in the path. Each device adds a small amount of processing delay.
3.  **Define the Target (Output Formula):** We will create a formula that combines these features with some randomness (noise) to simulate real-world variability.
    ```python
    import pandas as pd
    import numpy as np

    # Configure the data generation
    num_samples = 5000
    np.random.seed(42)

    # 1. Create the features
    data = {
        'distance_km': np.random.uniform(1, 500, num_samples),
        'traffic_load_gbps': np.random.uniform(1, 100, num_samples),
        'device_count': np.random.randint(2, 10, num_samples)
    }
    df = pd.DataFrame(data)

    # 2. Create the target variable 'latency_ms' using a formula
    # Latency from distance (speed of light in fiber is ~5 microseconds/km)
    base_latency = df['distance_km'] * 0.005 
    # Latency from device hops (e.g., 0.1ms per device)
    device_latency = df['device_count'] * 0.1
    # Latency from congestion (non-linear effect)
    load_effect = (df['traffic_load_gbps'] ** 1.2) * 0.01 
    # Add some random noise
    noise = np.random.normal(0, 0.5, num_samples)

    df['latency_ms'] = base_latency + device_latency + load_effect + noise
    # Ensure latency can't be negative
    df['latency_ms'] = df['latency_ms'].clip(lower=0)

    print(df.head())
    # Save for future use
    df.to_csv('synthetic_latency_data.csv', index=False)
    ```

**Step 3: Exploratory Data Analysis (EDA)**
1.  Visualize the relationships in your new dataset. A correlation heatmap is perfect for this.
    ```python
    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Feature Correlation Heatmap')
    plt.show()
    ```
    *You should see strong, positive correlations between all features and `latency_ms`, which confirms your formula worked as intended.*

**Step 4: Prepare Data for Modeling**
1.  Split the data into features (`X`) and the target variable (`y`).
2.  Create training and testing sets to ensure you evaluate the model on unseen data.
    ```python
    from sklearn.model_selection import train_test_split

    X = df.drop('latency_ms', axis=1)
    y = df['latency_ms']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    ```

**Step 5: Train and Evaluate Models**
1.  **Baseline Model (Linear Regression):** Always start with a simple model to set a performance benchmark.
    ```python
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error

    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
    print(f"Baseline Linear Regression RMSE: {rmse_lr:.2f} ms")
    ```
2.  **Advanced Model (XGBoost):** Train the high-performance gradient boosting model.
    ```python
    import xgboost as xgb

    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
    print(f"XGBoost Regressor RMSE: {rmse_xgb:.2f} ms")
    ```
    *The XGBoost RMSE should be significantly lower (better) than the baseline, because it can capture the non-linear `load_effect` we created.*

**Step 6: Visualize Results and Feature Importance**
1.  **Plot Predicted vs. Actual Values:** A scatter plot is the best way to visualize the performance of a regression model.
    ```python
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred_xgb, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Latency (ms)')
    plt.ylabel('Predicted Latency (ms)')
    plt.title('XGBoost: Actual vs. Predicted Latency')
    plt.show()
    ```
    *The closer the points hug the red diagonal line, the more accurate the model's predictions are.*

2.  **Plot Feature Importance:** Understand which factors the model found most predictive.
    ```python
    feature_importances = pd.DataFrame({'feature': X.columns, 'importance': xgb_model.feature_importances_})
    feature_importances = feature_importances.sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importances)
    plt.title('Feature Importance for Predicting Network Latency')
    plt.show()
    ```

#### **6. Success Criteria**
*   The team can successfully generate a synthetic dataset where the relationships between features and the target are logical and observable.
*   The XGBoost model demonstrates a significantly lower Root Mean Squared Error (RMSE) compared to the linear regression baseline.
*   The team can produce and correctly interpret the "Actual vs. Predicted" scatter plot.
*   The team can generate and interpret the feature importance plot, explaining which simulated factors had the largest impact on the model's predictions.

#### **7. Next Steps & Extensions**
*   **Add Categorical Features:** Modify the data generation script to include categorical features like `vendor_type` or `path_redundancy` ('Yes'/'No'). This will require the team to practice one-hot encoding within the modeling pipeline.
*   **Hyperparameter Tuning:** Use `GridSearchCV` or `RandomizedSearchCV` from Scikit-learn to find the optimal hyperparameters for the `XGBRegressor`, which can further improve its accuracy.
*   **Model Deployment:** Save the trained XGBoost model to a file using `joblib`. Write a simple Python script that loads the model and allows a user to input feature values (distance, load, etc.) to get a real-time latency prediction.

---