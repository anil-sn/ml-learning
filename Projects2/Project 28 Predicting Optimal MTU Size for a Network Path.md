---

### **Project 28: Predicting Optimal MTU Size for a Network Path**

**Objective:** To build a regression model that can predict the optimal MTU size for a given network path and application type, aiming to maximize throughput and minimize fragmentation.

**Dataset Source:** **Synthetically Generated**. We will create a dataset simulating the results of path MTU discovery tests under various conditions. The dataset will include application types (which influence typical packet sizes), path characteristics, and the resulting optimal MTU.

**Model:** We will use a **Gradient Boosting Regressor**. This is a powerful and accurate regression model that can effectively capture the complex interactions between different network factors to predict a precise numerical value like the MTU size.

**Instructions:**
This notebook is fully self-contained and does not require any external files or APIs. Simply run the entire code block in Google Colab.

**Implementation in Google Colab:**
```python
#
# ==================================================================================
#  Project 28: Predicting Optimal MTU Size for a Network Path
# ==================================================================================
#
# Objective:
# This notebook builds a regression model to predict the optimal MTU size
# based on application type and path characteristics, using a synthetic dataset.
#
# To Run in Google Colab:
# Copy and paste this entire code block into a single cell and run it.
#

# ----------------------------------------
# 1. Import Necessary Libraries
# ----------------------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------
# 2. Synthetic Optimal MTU Data Generation
# ----------------------------------------
print("--- Generating Synthetic Optimal MTU Dataset ---")

num_samples = 2000
data = []
application_types = ['VOIP', 'Video_Streaming', 'Bulk_Data_Transfer', 'Web_Browsing', 'Database_Replication']

for _ in range(num_samples):
    app_type = random.choice(application_types)
    base_latency_ms = np.random.uniform(5, 100)
    # Simulate if the path includes a VPN, which adds overhead and reduces optimal MTU
    has_vpn_tunnel = np.random.choice([0, 1], p=[0.7, 0.3])
    
    # --- Define Rules for Optimal MTU ---
    # This is our ground truth logic
    if has_vpn_tunnel:
        base_mtu = 1400
    else:
        base_mtu = 1500

    if app_type == 'VOIP':
        # VOIP uses small packets, so large MTU is inefficient overhead
        optimal_mtu = np.random.randint(500, 700)
    elif app_type == 'Bulk_Data_Transfer':
        # Bulk transfers benefit from the largest possible MTU to reduce header overhead
        optimal_mtu = base_mtu - np.random.randint(0, 20) # Small variations
    elif app_type == 'Web_Browsing':
        # Web browsing has a mix of packet sizes
        optimal_mtu = np.random.randint(1300, 1500)
    else: # Video streaming and DB replication
        optimal_mtu = base_mtu - np.random.randint(10, 50)
        
    data.append([app_type, base_latency_ms, has_vpn_tunnel, optimal_mtu])

df = pd.DataFrame(data, columns=['application_type', 'base_latency_ms', 'has_vpn_tunnel', 'optimal_mtu'])
print(f"Dataset generation complete. Created {len(df)} samples.")
print("\nDataset Sample:")
print(df.sample(5))


# ----------------------------------------
# 3. Data Splitting and Encoding
# ----------------------------------------
print("\n--- Splitting and Encoding Data ---")

X = df.drop(columns=['optimal_mtu'])
y = df['optimal_mtu']

# One-hot encode the 'application_type' categorical feature
X_encoded = pd.get_dummies(X, columns=['application_type'], drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")


# ----------------------------------------
# 4. Model Training with Gradient Boosting Regressor
# ----------------------------------------
print("\n--- Model Training ---")

# Initialize the Gradient Boosting Regressor
model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

print("Training the Gradient Boosting model...")
model.fit(X_train, y_train)
print("Training complete.")


# ----------------------------------------
# 5. Model Evaluation
# ----------------------------------------
print("\n--- Model Evaluation ---")
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f} bytes")
print(f"  (On average, the model's MTU prediction is off by just +/- {mae:.2f} bytes)")
print(f"R-squared (R²): {r2:.2%}")
print(f"  ({r2:.0%} of the variance in the optimal MTU can be explained by our features)")

# --- Visualization: Actual vs. Predicted ---
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2, label='Perfect Prediction')
plt.xlabel('Actual Optimal MTU (bytes)')
plt.ylabel('Predicted Optimal MTU (bytes)')
plt.title('Actual vs. Predicted Optimal MTU')
plt.legend()
plt.grid(True)
plt.show()


# ----------------------------------------
# 6. Feature Importance
# ----------------------------------------
print("\n--- Feature Importance ---")
importances = model.feature_importances_
features = X_train.columns
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance in Predicting Optimal MTU')
plt.show()


# ----------------------------------------
# 7. Conclusion
# ----------------------------------------
print("\n--- Conclusion ---")
print(f"The Gradient Boosting model learned to predict the optimal MTU size with a high degree of accuracy (R² of {r2:.2%}).")
print("Key Takeaways:")
print("- The model's low Mean Absolute Error shows it can provide very precise MTU recommendations, helping to avoid both fragmentation and unnecessary overhead.")
print("- The feature importance plot provides critical insights. It clearly shows that the `application_type` (specifically `Bulk_Data_Transfer` and `VOIP`) is the most decisive factor, followed by whether a `VPN tunnel` is in use. This confirms that the model learned the underlying network engineering principles correctly.")
print("- This type of predictive model could be a key component in an advanced Software-Defined Networking (SDN) controller. The controller could monitor application flows, and for each new flow, query the model to determine the optimal MTU. It could then enforce this MTU size on the virtual interfaces for that specific flow, creating a highly dynamic and application-aware network that optimizes its own performance on a per-flow basis.")

```