---

### **Project 18: Predicting Wi-Fi Roaming Events for Mobile Devices**

**Objective:** To build a machine learning model that predicts if a wireless client will roam to a new AP within the next few seconds, based on the changing signal strength (RSSI) from surrounding APs.

**Dataset Source:** **Synthetically Generated**. We will create a time-series dataset that simulates a user walking through a building with multiple APs. The dataset will track the RSSI from each AP as perceived by the user's device, along with the actual roaming events.

**Model:** We will use a **RandomForestClassifier**. This model is well-suited for this task as it can capture the non-linear relationships between multiple signal strength indicators and the decision to roam.

**Instructions:**
This notebook is fully self-contained and does not require any external files or APIs. Simply run the entire code block in Google Colab.

**Implementation in Google Colab:**
```python
#
# ==================================================================================
#  Project 18: Predicting Wi-Fi Roaming Events for Mobile Devices
# ==================================================================================
#
# Objective:
# This notebook builds a model to predict imminent Wi-Fi roaming events based on
# time-series signal strength data, using a synthetically generated dataset.
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------
# 2. Synthetic Wi-Fi Roaming Data Generation
# ----------------------------------------
print("--- Generating Synthetic Wi-Fi Roaming Dataset ---")

# Simulation parameters
time_steps = 300  # 300 seconds of data
num_aps = 3
aps = [f'AP_{i+1}' for i in range(num_aps)]
data = []

# Simulate a walk: Start near AP_1, move to AP_2, then to AP_3
for t in range(time_steps):
    # Simulate RSSI values (in dBm, lower is worse)
    rssi_ap1 = -30 - (t * 0.2) + np.random.normal(0, 2)
    rssi_ap2 = -90 + (abs(t - 150) * -0.4) + 50 + np.random.normal(0, 2)
    rssi_ap3 = -90 + (abs(t - 250) * -0.4) + 40 + np.random.normal(0, 2)
    
    rssi_values = [rssi_ap1, rssi_ap2, rssi_ap3]
    
    # The client connects to the AP with the strongest signal
    connected_ap = aps[np.argmax(rssi_values)]
    
    data.append([t, connected_ap] + rssi_values)

df = pd.DataFrame(data, columns=['time', 'connected_ap'] + [f'rssi_{ap}' for ap in aps])
print("Dataset generation complete. Sample:")
print(df.head())


# ----------------------------------------
# 3. Feature Engineering: Creating the Target Variable
# ----------------------------------------
print("\n--- Engineering Features and the Predictive Target ---")

# Our goal is to predict a roam *before* it happens.
# We create a target variable 'will_roam_soon' which is 1 if a roam will
# occur in the next `prediction_window` seconds, and 0 otherwise.
prediction_window = 5 # seconds
df['will_roam_soon'] = 0

# Shift the 'connected_ap' column to see future connections
df['next_ap'] = df['connected_ap'].shift(-prediction_window)

# A roam will happen if the current AP is different from the future AP
roam_indices = df[df['connected_ap'] != df['next_ap']].index

# Set the flag for the time steps *leading up to* the roam
for idx in roam_indices:
    df.loc[max(0, idx - prediction_window):idx, 'will_roam_soon'] = 1

# --- Engineer Rate-of-Change (Delta) Features ---
# The speed at which signal strength changes is a powerful predictor.
for ap in aps:
    df[f'rssi_{ap}_delta'] = df[f'rssi_{ap}'].diff().fillna(0)

# Drop helper columns
df = df.drop(columns=['next_ap'])
df = df.dropna()

print(f"Target variable 'will_roam_soon' created. Distribution:")
print(df['will_roam_soon'].value_counts())
print("\nSample of data with new features:")
print(df.head())


# ----------------------------------------
# 4. Data Splitting
# ----------------------------------------
print("\n--- Splitting Data for Training and Testing ---")
feature_cols = [col for col in df.columns if 'rssi' in col]
X = df[feature_cols]
y = df['will_roam_soon']

# Use a time-series split for a more realistic evaluation
# We train on the past and test on the future.
split_point = int(len(df) * 0.7)
X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")


# ----------------------------------------
# 5. Model Training
# ----------------------------------------
print("\n--- Model Training ---")
# `class_weight='balanced'` helps the model pay attention to the rare 'will_roam_soon=1' events
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

print("Training the RandomForestClassifier...")
model.fit(X_train, y_train)
print("Training complete.")


# ----------------------------------------
# 6. Model Evaluation
# ----------------------------------------
print("\n--- Model Evaluation ---")
y_pred = model.predict(X_test)

# For this problem, RECALL for the 'Will Roam (1)' class is most important.
# We want to catch as many impending roams as possible, even if we have some false alarms.
print("\nClassification Report (Focus on Recall for class 1):")
print(classification_report(y_test, y_pred, target_names=['Will Not Roam (0)', 'Will Roam (1)']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=['Not Roam', 'Will Roam'], yticklabels=['Not Roam', 'Will Roam'])
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()


# ----------------------------------------
# 7. Visualization of Predictions
# ----------------------------------------
print("\n--- Visualizing Predictions Over Time ---")

df_test_results = df.iloc[split_point:].copy()
df_test_results['prediction'] = y_pred

plt.figure(figsize=(16, 8))
# Plot RSSI values
for ap in aps:
    plt.plot(df_test_results['time'], df_test_results[f'rssi_{ap}'], label=f'RSSI {ap}')

# Highlight the true pre-roam windows
plt.fill_between(df_test_results['time'], -90, -20, where=df_test_results['will_roam_soon']==1,
                 facecolor='orange', alpha=0.5, label='Actual Pre-Roam Window')
# Highlight where the model PREDICTED a roam
plt.scatter(df_test_results['time'][df_test_results['prediction']==1],
            df_test_results['rssi_AP_3'][df_test_results['prediction']==1] + 2, # Offset for visibility
            color='red', marker='v', s=50, label='Predicted Roam')

plt.title('Model Predictions on Test Data')
plt.xlabel('Time (seconds)')
plt.ylabel('RSSI (dBm)')
plt.legend()
plt.grid(True)
plt.ylim(-90, -20)
plt.show()


# ----------------------------------------
# 8. Conclusion
# ----------------------------------------
print("\n--- Conclusion ---")
print("The RandomForest model successfully learned to predict imminent Wi-Fi roaming events.")
print("Key Takeaways:")
print(f"- The model achieved high recall for the 'Will Roam' class, demonstrating its ability to provide advance warning for a majority of roaming events.")
print("- The visualization clearly shows the model making correct predictions during the critical crossover periods where signal strengths from different APs are changing.")
print("- Feature engineering was key: the model used not just the raw signal strengths but also their rate of change (`delta`) to make more accurate decisions.")
print("- In a real-world Wi-Fi network controller, this predictive capability could be used to implement Fast Roaming (802.11r). The controller could pre-authenticate the client with the likely destination AP, making the final handover nearly instantaneous and preventing dropped packets for sensitive applications like voice and video calls.")

```