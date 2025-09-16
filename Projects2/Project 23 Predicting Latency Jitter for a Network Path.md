---

### **Project 23: Predicting Latency/Jitter for a Network Path**

**Objective:** To build a regression model that can predict the latency (in milliseconds) of a network path based on characteristics like distance, time of day, and current traffic load.

**Dataset Source:** **Synthetically Generated**. We will create a realistic dataset that simulates network monitoring probes. This dataset will include various operational factors that influence latency, allowing us to train a model to understand these complex relationships.

**Model:** We will use the **XGBoost Regressor**. XGBoost is a state-of-the-art gradient boosting library that is highly effective for tabular regression tasks, capable of capturing non-linear relationships and feature interactions to produce highly accurate predictions.

**Instructions:**
This notebook is fully self-contained and does not require any external files or APIs. Simply run the entire code block in Google Colab.

**Implementation in Google Colab:**
```python
#
# ==================================================================================
#  Project 23: Predicting Latency/Jitter for a Network Path
# ==================================================================================
#
# Objective:
# This notebook builds a regression model to predict network latency based on
# path characteristics, using a synthetically generated dataset.
#
# To Run in Google Colab:
# Copy and paste this entire code block into a single cell and run it.
#

# ----------------------------------------
# 1. Import Necessary Libraries
# ----------------------------------------
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------
# 2. Synthetic Latency Data Generation
# ----------------------------------------
print("--- Generating Synthetic Network Latency Dataset ---")

num_samples = 5000
data = []

for _ in range(num_samples):
    distance_km = np.random.randint(50, 5000)
    hour_of_day = np.random.randint(0, 24)
    
    # Simulate business hours congestion (9am to 5pm)
    if 9 <= hour_of_day <= 17:
        congestion_factor = np.random.uniform(1.2, 2.5) # Higher latency during these hours
    else:
        congestion_factor = np.random.uniform(0.8, 1.2) # Lower latency off-hours
        
    # Simulate random traffic spikes
    traffic_spike = np.random.choice([0, 1], p=[0.9, 0.1]) * np.random.uniform(5, 15)
    
    # --- Create a realistic formula for latency ---
    # Baseline latency + distance effect + congestion effect + random noise
    base_latency = 5 # ms for local processing
    distance_latency = distance_km * 0.05 # rough speed-of-light factor
    congestion_latency = hour_of_day * congestion_factor
    random_noise = np.random.normal(0, 5) # Simulates jitter
    
    # Final latency
    latency = base_latency + distance_latency + congestion_latency + traffic_spike + random_noise
    latency = max(5, latency) # Ensure latency is not unrealistically low
    
    data.append([distance_km, hour_of_day, congestion_factor, traffic_spike, latency])

df = pd.DataFrame(data, columns=['distance_km', 'hour_of_day', 'congestion_factor', 'traffic_spike', 'latency_ms'])

print("Dataset generation complete. Sample:")
print(df.sample(5))


# ----------------------------------------
# 3. Exploratory Data Analysis (EDA)
# ----------------------------------------
print("\n--- Visualizing Feature Relationships ---")

sns.pairplot(df, x_vars=['distance_km', 'hour_of_day', 'congestion_factor'], y_vars=['latency_ms'], height=4, aspect=1)
plt.suptitle('Latency vs. Key Features', y=1.02)
plt.show()
print("The plots show a strong positive correlation between distance and latency, and a more complex relationship between the hour of day and latency.")


# ----------------------------------------
# 4. Data Splitting
# ----------------------------------------
print("\n--- Splitting Data for Training and Testing ---")

feature_cols = ['distance_km', 'hour_of_day', 'congestion_factor', 'traffic_spike']
X = df[feature_cols]
y = df['latency_ms']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")


# ----------------------------------------
# 5. Model Training with XGBoost Regressor
# ----------------------------------------
print("\n--- Model Training ---")

# Initialize the XGBoost Regressor
model = xgb.XGBRegressor(
    objective='reg:squarederror', # Objective function for regression
    n_estimators=100,             # Number of boosting rounds
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1
)

print("Training the XGBoost Regressor model...")
model.fit(X_train, y_train)
print("Training complete.")


# ----------------------------------------
# 6. Model Evaluation for Regression
# ----------------------------------------
print("\n--- Model Evaluation ---")
y_pred = model.predict(X_test)

# --- Key Regression Metrics ---
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE):  {mae:.2f} ms")
print(f"  (On average, the model's prediction is off by +/- {mae:.2f} ms)")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} ms")
print(f"  (Penalizes larger errors more heavily)")
print(f"R-squared (R²): {r2:.2%}")
print(f"  ({r2:.0%} of the variance in latency can be explained by our features)")

# --- Visualization: Actual vs. Predicted ---
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2, label='Perfect Prediction')
plt.xlabel('Actual Latency (ms)')
plt.ylabel('Predicted Latency (ms)')
plt.title('Actual vs. Predicted Latency')
plt.legend()
plt.grid(True)
plt.show()


# ----------------------------------------
# 7. Feature Importance
# ----------------------------------------
print("\n--- Feature Importance ---")
fig, ax = plt.subplots(figsize=(10, 6))
xgb.plot_importance(model, ax=ax, height=0.8)
plt.title('Feature Importance in Predicting Latency')
plt.show()


# ----------------------------------------
# 8. Conclusion
# ----------------------------------------
print("\n--- Conclusion ---")
print(f"The XGBoost Regressor model successfully learned to predict network latency with high accuracy, achieving an R-squared score of {r2:.2%}.")
print("Key Takeaways:")
print(f"- The model's Mean Absolute Error of {mae:.2f} ms shows it can produce reliable, actionable latency estimates for network monitoring.")
print("- The 'Actual vs. Predicted' plot confirms the model's strong performance, as most predictions lie very close to the ideal 'Perfect Prediction' line.")
print("- The feature importance plot provides crucial insights for network engineers. It confirms that `distance_km` is the single most dominant factor in latency, followed by the time of day (`hour_of_day`) and its associated congestion.")
print("- A model like this could be deployed in a Network Operations Center (NOC) to power a 'what-if' analysis tool. An operator could ask, 'What will the latency to our new London data center (3400 km away) be during peak business hours?' and get an instant, data-driven estimate, improving planning and design decisions.")

```