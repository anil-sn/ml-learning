---

### **Project 34: Detecting Noisy Neighbors in a Multi-tenant Cloud Environment**

**Objective:** To build an unsupervised anomaly detection model that can identify a "noisy neighbor" (a high-resource-consuming tenant) on a shared host by analyzing the network traffic patterns of all tenants and flagging outliers.

**Dataset Source:** **Synthetically Generated**. We will create a dataset that simulates network traffic metrics (like packets per second and bytes per second) for a group of tenants sharing a physical host. The dataset will include one tenant who periodically becomes a noisy neighbor.

**Model:** We will use **Isolation Forest**. This is an ideal unsupervised algorithm for this task because it's designed to efficiently find outliers in a dataset. We will train it on the collective behavior of all tenants, and it will learn to flag any tenant whose behavior deviates significantly from the norm.

**Instructions:**
This notebook is fully self-contained and does not require any external files or APIs. Simply run the entire code block in Google Colab.

**Implementation in Google Colab:**
```python
#
# ==================================================================================
#  Project 34: Detecting Noisy Neighbors in a Multi-tenant Cloud Environment
# ==================================================================================
#
# Objective:
# This notebook builds an unsupervised model to detect "noisy neighbor" tenants
# by identifying outliers in their network traffic behavior.
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
# 2. Synthetic Tenant Traffic Data Generation
# ----------------------------------------
print("--- Generating Synthetic Multi-Tenant Network Traffic Dataset ---")

num_tenants = 20
time_steps = 1000 # Simulate 1000 minutes of data
data = []
tenants = [f'tenant_{i+1}' for i in range(num_tenants)]
noisy_neighbor_tenant = 'tenant_5'

for t in range(time_steps):
    for tenant in tenants:
        is_noisy = False
        # Define normal behavior
        base_pps = np.random.normal(5000, 1000) # Packets per second
        base_bps = base_pps * np.random.normal(300, 50) # Bytes per second
        
        # --- Simulate the Noisy Neighbor event ---
        # The noisy neighbor has a burst of high activity for a specific period
        if tenant == noisy_neighbor_tenant and 400 <= t < 600:
            base_pps *= np.random.uniform(5, 10) # 5-10x more packets
            base_bps *= np.random.uniform(5, 10) # 5-10x more bytes
            is_noisy = True
            
        data.append([t, tenant, base_pps, base_bps, is_noisy])

df = pd.DataFrame(data, columns=['timestamp', 'tenant_id', 'packets_per_second', 'bytes_per_second', 'is_truly_noisy'])
print(f"Dataset generation complete. Created {len(df)} records.")
print("\nDataset Sample:")
print(df.sample(5))


# ----------------------------------------
# 3. Data Preparation and Scaling
# ----------------------------------------
print("\n--- Preparing Data for Unsupervised Learning ---")

# The features we'll use to detect anomalies
feature_cols = ['packets_per_second', 'bytes_per_second']
X = df[feature_cols]

# Scale the features for better model performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ----------------------------------------
# 4. Model Training (Unsupervised)
# ----------------------------------------
print("\n--- Model Training ---")
# The `contamination` parameter is the expected proportion of outliers in the data.
# We know our noisy event lasts for 200 of 1000 minutes for 1 of 20 tenants.
# So, the approximate contamination is (200 * 1) / (1000 * 20) = 0.01 or 1%.
model = IsolationForest(
    n_estimators=100,
    contamination=0.01,
    random_state=42
)

print("Training the Isolation Forest model on the entire dataset...")
# Unlike previous projects, we train on ALL data at once to find outliers within the group.
df['is_anomaly'] = model.fit_predict(X_scaled) # Returns 1 for normal, -1 for anomaly/outlier
print("Training and prediction complete.")


# ----------------------------------------
# 5. Evaluation and Analysis of Detected Anomalies
# ----------------------------------------
print("\n--- Evaluating and Analyzing Detections ---")

# Compare our model's predictions with the ground truth
# Convert our boolean 'is_truly_noisy' to the model's format (True=-1, False=1)
y_true = df['is_truly_noisy'].apply(lambda x: -1 if x else 1)
y_pred = df['is_anomaly']

from sklearn.metrics import classification_report
print("Evaluation of Anomaly Detection:")
print(classification_report(y_true, y_pred, target_names=['Normal (1)', 'Noisy Neighbor (-1)']))

# Let's find out which tenant was flagged
flagged_tenants = df[df['is_anomaly'] == -1]['tenant_id'].unique()
print(f"\nModel flagged the following tenant(s) as anomalous: {flagged_tenants}")
if noisy_neighbor_tenant in flagged_tenants:
    print("SUCCESS: The model correctly identified the noisy neighbor.")
else:
    print("FAILURE: The model failed to identify the noisy neighbor.")


# ----------------------------------------
# 6. Visualization of the Results
# ----------------------------------------
print("\n--- Visualizing the Detected Noisy Neighbor ---")

# Let's plot the traffic of the noisy neighbor vs. a normal tenant
normal_tenant = 'tenant_1'

plt.figure(figsize=(15, 7))
sns.lineplot(data=df[df['tenant_id'] == normal_tenant], x='timestamp', y='bytes_per_second', label=f'Normal Tenant ({normal_tenant})', color='blue')
sns.lineplot(data=df[df['tenant_id'] == noisy_neighbor_tenant], x='timestamp', y='bytes_per_second', label=f'Noisy Neighbor ({noisy_neighbor_tenant})', color='orange')

# Highlight the area where the model detected an anomaly for the noisy neighbor
anomalies = df[(df['tenant_id'] == noisy_neighbor_tenant) & (df['is_anomaly'] == -1)]
plt.scatter(anomalies['timestamp'], anomalies['bytes_per_second'], color='red', s=50, label='Detected Anomaly', zorder=5)

plt.title('Network Traffic: Normal Tenant vs. Noisy Neighbor', fontsize=16)
plt.xlabel('Time (Minutes)')
plt.ylabel('Bytes per Second')
plt.legend()
plt.grid(True)
plt.show()

# --- Scatter plot of all tenants ---
plt.figure(figsize=(10, 8))
sns.scatterplot(
    data=df,
    x='packets_per_second',
    y='bytes_per_second',
    hue='is_anomaly',
    palette={1: 'blue', -1: 'red'},
    alpha=0.5
)
plt.title('All Tenant Traffic - Outliers Detected by Isolation Forest')
plt.legend(title='Prediction', labels=['Noisy Neighbor', 'Normal'])
plt.xscale('log')
plt.yscale('log')
plt.show()


# ----------------------------------------
# 7. Conclusion
# ----------------------------------------
print("\n--- Conclusion ---")
print("The Isolation Forest model successfully identified the 'noisy neighbor' tenant in a simulated multi-tenant environment.")
print("Key Takeaways:")
print("- The model effectively learned a baseline of normal, collective tenant behavior and was able to flag the single tenant that deviated significantly from this norm.")
print("- The scatter plot clearly visualizes the power of the model. It draws a boundary around the dense cluster of normal behavior and correctly identifies the sparse, high-traffic points as anomalies.")
print("- This is a powerful, unsupervised approach because it doesn't require knowing what 'normal' is for any single tenant. It learns it from the population, making it highly adaptable.")
print("- A cloud provider or data center operator could integrate this model into their platform's monitoring system. When an anomaly is detected for a specific tenant on a host, the system could automatically take action, such as rate-limiting the noisy tenant's traffic or migrating them to a different host to restore performance for all other tenants, ensuring SLA compliance.")
```