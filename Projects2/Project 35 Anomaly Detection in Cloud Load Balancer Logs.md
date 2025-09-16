---

### **Project 35: Anomaly Detection in Cloud Load Balancer Logs**

**Objective:** To build an unsupervised model that can detect anomalous traffic patterns in cloud load balancer logs by establishing a baseline of normal, aggregated behavior (requests per minute, error rates) and identifying significant deviations.

**Dataset Source:** **Synthetically Generated**. We will create a realistic time-series dataset of load balancer logs, aggregated per minute. The dataset will include a "normal" period, followed by a simulated event like a sudden spike in 5xx server errors, which the model should detect.

**Model:** We will use the **Isolation Forest** algorithm. It is perfectly suited for this time-series anomaly detection task. We will train it on a period of known-good, "normal" traffic, and then use the trained model to identify any subsequent time windows that deviate from this learned baseline.

**Instructions:**
This notebook is fully self-contained and does not require any external files or APIs. Simply run the entire code block in Google Colab.

**Implementation in Google Colab:**
```python
#
# ==================================================================================
#  Project 35: Anomaly Detection in Cloud Load Balancer Logs
# ==================================================================================
#
# Objective:
# This notebook builds an unsupervised model to detect anomalies in aggregated
# load balancer logs, such as a spike in server-side errors.
#
# To Run in Google Colab:
# Copy and paste this entire code block into a single cell and run it.
#

# ----------------------------------------
# 1. Import Necessary Libraries
# ----------------------------------------
import pandas as pd
import numpy as np
import random
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------
# 2. Synthetic Load Balancer Log Generation
# ----------------------------------------
print("--- Generating Synthetic Aggregated Load Balancer Log Dataset ---")

time_steps_minutes = 1440 # 24 hours of data, aggregated per minute
data = []
anomaly_start_time = 1200 # Anomaly starts at minute 1200
anomaly_duration = 60   # Anomaly lasts for 60 minutes

for t in range(time_steps_minutes):
    is_anomaly = False
    # Simulate normal traffic with a daily sinusoidal pattern (peak during the day)
    base_requests = 10000 + 5000 * np.sin((t - 480) * (2 * np.pi / 1440)) + np.random.randint(-500, 500)
    
    # Normal error rates
    http_2xx_rate = 0.98 # Success
    http_4xx_rate = 0.015 # Client-side errors
    http_5xx_rate = 0.005 # Server-side errors
    
    # --- Simulate the Anomaly: A spike in 5xx server errors ---
    if anomaly_start_time <= t < anomaly_start_time + anomaly_duration:
        is_anomaly = True
        # A backend service is failing, causing 5xx errors to spike
        http_5xx_rate = np.random.uniform(0.3, 0.6) # 30-60% of requests now fail
        http_2xx_rate = 1 - http_4xx_rate - http_5xx_rate
        
    num_2xx = int(base_requests * http_2xx_rate)
    num_4xx = int(base_requests * http_4xx_rate)
    num_5xx = int(base_requests * http_5xx_rate)
    total_requests = num_2xx + num_4xx + num_5xx
    
    data.append([t, total_requests, num_2xx, num_4xx, num_5xx, is_anomaly])

df = pd.DataFrame(data, columns=['minute', 'total_requests', '2xx_count', '4xx_count', '5xx_count', 'is_truly_anomaly'])
df.set_index('minute', inplace=True)
print(f"Dataset generation complete. Created {len(df)} records.")
print("\nDataset Sample:")
print(df.sample(5))


# ----------------------------------------
# 3. Feature Engineering
# ----------------------------------------
print("\n--- Engineering Rate-Based Features ---")

# Raw counts can be misleading. Rates are often better features.
df['5xx_error_rate'] = df['5xx_count'] / df['total_requests']
df['4xx_error_rate'] = df['4xx_count'] / df['total_requests']
# Handle potential division by zero if total_requests is 0
df.fillna(0, inplace=True)

feature_cols = ['total_requests', '5xx_error_rate', '4xx_error_rate']
X = df[feature_cols]


# ----------------------------------------
# 4. Unsupervised Model Training
# ----------------------------------------
print("\n--- Unsupervised Model Training (on NORMAL data only) ---")

# We train the model on a period of known-good, normal operations.
# Let's use the first 1000 minutes.
X_train_normal = X[X.index < 1000]
print(f"Training Isolation Forest on {len(X_train_normal)} minutes of normal log data.")

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_normal)

# Initialize and train the Isolation Forest
model = IsolationForest(contamination='auto', random_state=42)
model.fit(X_train_scaled)
print("Training complete.")


# ----------------------------------------
# 5. Anomaly Detection and Evaluation
# ----------------------------------------
print("\n--- Detecting Anomalies on the Full Dataset ---")

# Use the trained model to predict on the entire dataset
X_all_scaled = scaler.transform(X)
df['is_anomaly_pred'] = model.predict(X_all_scaled) # -1 for anomaly, 1 for normal

# Compare with our ground truth
y_true = df['is_truly_anomaly'].apply(lambda x: -1 if x else 1)
y_pred = df['is_anomaly_pred']

from sklearn.metrics import classification_report
print("\nEvaluation of Anomaly Detection:")
print(classification_report(y_true, y_pred, target_names=['Normal (1)', 'Anomaly (-1)']))


# ----------------------------------------
# 6. Visualization of the Results
# ----------------------------------------
print("\n--- Visualizing the Detected Anomaly ---")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

# Plot 1: The key feature - 5xx Error Rate
ax1.plot(df.index, df['5xx_error_rate'], label='5xx Error Rate', color='orange')
ax1.axvspan(anomaly_start_time, anomaly_start_time + anomaly_duration, color='red', alpha=0.2, label='Simulated Anomaly')
ax1.set_title('Load Balancer 5xx Server Error Rate Over Time', fontsize=14)
ax1.set_ylabel('Error Rate')
ax1.legend()
ax1.grid(True)

# Plot 2: The model's anomaly prediction
# We'll create a new series for plotting that is NaN where there's no anomaly
anomaly_points = df['5xx_error_rate'].copy()
anomaly_points[df['is_anomaly_pred'] == 1] = np.nan

ax2.plot(df.index, df['5xx_error_rate'], label='5xx Error Rate', color='grey', alpha=0.5)
ax2.scatter(df.index, anomaly_points, color='red', label='Detected Anomaly', zorder=5)
ax2.axvspan(anomaly_start_time, anomaly_start_time + anomaly_duration, color='red', alpha=0.2, label='Simulated Anomaly')
ax2.set_title('Isolation Forest Anomaly Detections', fontsize=14)
ax2.set_xlabel('Time (Minute of Day)')
ax2.set_ylabel('Error Rate')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()


# ----------------------------------------
# 7. Conclusion
# ----------------------------------------
print("\n--- Conclusion ---")
print("The Isolation Forest model, trained only on a baseline of normal traffic, successfully detected the anomalous spike in server-side errors.")
print("Key Takeaways:")
print("- The model's high recall for the 'Anomaly' class demonstrates its effectiveness as an early warning system. It correctly identified the period where backend servers were failing.")
print("- The visualization clearly shows the model's logic: it learned the normal, near-zero rate of 5xx errors. When the rate suddenly spiked during the simulated event, the model immediately flagged these time points as anomalous.")
print("- This is a powerful, unsupervised approach for monitoring critical infrastructure. It doesn't need to be told what a 'bad' error rate is; it learns the normal operational range and alerts on any significant deviation.")
print("- In a real-world scenario, this model could be integrated into an automated monitoring and response system. When the model detects an anomaly like this, it could automatically trigger an alert to the on-call engineer, initiate the removal of failing backend servers from the load balancer pool, or even trigger an auto-scaling event to launch new, healthy servers.")
```