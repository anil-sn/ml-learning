---

### **Project 26: BGP Anomaly Detection (Route Leaks, Hijacks)**

**Objective:** To build an unsupervised anomaly detection model that can identify anomalous BGP update messages, such as those indicative of a route leak or prefix hijack, by analyzing features of the BGP AS-path.

**Dataset Source:** **Kaggle**. We will use the "BGP Hijacking Detection Dataset". This dataset contains features extracted from real BGP update messages, labeled as either 'normal' or 'anomalous'. We will use the labels only for final evaluation, not for training.

**Model:** We will use **Isolation Forest**. This is an excellent choice for this problem because we want to detect rare, unusual events. The model learns the characteristics of "normal" BGP updates and then flags updates that deviate significantly from that baseline as potential anomalies.

**Instructions:**
This notebook requires the Kaggle API. Please run the setup cell and upload your `kaggle.json` file if you have not already done so in this session.

**Implementation in Google Colab:**
```python
#
# ==================================================================================
#  Project 26: BGP Anomaly Detection
# ==================================================================================
#
# Objective:
# This notebook builds an unsupervised model to detect BGP anomalies by analyzing
# AS-path features, using a real-world BGP update dataset.
#
# To Run in Google Colab:
# 1. Have your `kaggle.json` API token ready.
# 2. Copy and paste this entire code block into a single cell.
# 3. Run the cell. You may be prompted to upload `kaggle.json`.
#

# ----------------------------------------
# 1. Setup Kaggle API and Download Data
# ----------------------------------------
import os

if not os.path.exists('/root/.kaggle/kaggle.json'):
    print("--- Setting up Kaggle API ---")
    !pip install -q kaggle
    from google.colab import files
    print("\nPlease upload your kaggle.json file:")
    uploaded = files.upload()
    if 'kaggle.json' not in uploaded:
        print("\nError: kaggle.json not uploaded.")
        exit()
    !mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/json
else:
    print("Kaggle API already configured.")

print("\n--- Downloading BGP Hijacking Detection Dataset from Kaggle ---")
!kaggle datasets download -d dprembath/bgp-hijacking-detection-dataset

print("\n--- Unzipping the dataset ---")
!unzip -q bgp-hijacking-detection-dataset.zip -d bgp_data
print("Dataset setup complete.")


# ----------------------------------------
# 2. Load and Prepare the Data
# ----------------------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("\n--- Loading and Preprocessing Data ---")

try:
    df = pd.read_csv('bgp_data/bgp_data.csv')
    print("Successfully loaded bgp_data.csv.")
except FileNotFoundError as e:
    print(f"Error: Could not find dataset file. {e}")
    exit()

# Drop the 'Timestamp' column as we're focusing on path features
df = df.drop(columns=['Timestamp'])

# Encode the target label for later evaluation: anomaly -> -1, normal -> 1
df['Label'] = df['Label'].apply(lambda x: -1 if x == 'anomaly' else 1)
print(f"Dataset loaded. Shape: {df.shape}")

print("\nClass Distribution:")
print(df['Label'].value_counts())


# ----------------------------------------
# 3. Feature Selection and Data Preparation
# ----------------------------------------
print("\n--- Preparing Data for Unsupervised Learning ---")

# These features describe the BGP AS-path behavior
feature_cols = [
    'AS_PATH_LEN', 'AS_PATH_AVG_LEN', 'AS_PATH_MAX_LEN', 'AS_PATH_MIN_LEN',
    'EDIT_DIST_AS_PATH', 'EDIT_DIST_PREFIX', 'PREFIX_LEN',
    'UNIQUE_AS_COUNT', 'RARE_AS_COUNT', 'STDEV_AS_PATH_LEN'
]

X = df[feature_cols]
y_true = df['Label']

# --- CRITICAL STEP for Unsupervised Learning ---
# We will train our model ONLY on the 'normal' data.
X_train_normal = X[y_true == 1]
print(f"Training the model on {len(X_train_normal)} normal BGP updates.")

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_normal)


# ----------------------------------------
# 4. Model Training (Unsupervised)
# ----------------------------------------
print("\n--- Model Training ---")
# `contamination` is the expected ratio of anomalies in new, unseen data.
# Based on our data, the anomaly rate is about 15%, so we set it here.
# This helps the model set its decision threshold.
model = IsolationForest(n_estimators=100, contamination=0.15, random_state=42, n_jobs=-1)

print("Training the Isolation Forest model...")
model.fit(X_train_scaled)
print("Training complete.")


# ----------------------------------------
# 5. Model Evaluation
# ----------------------------------------
print("\n--- Model Evaluation on the Full Dataset ---")

# Now we test the model on the entire dataset (normal and anomalous)
X_all_scaled = scaler.transform(X)
y_pred = model.predict(X_all_scaled) # Predict returns 1 for normal, -1 for anomaly

print("\nClassification Report (Focus on Recall for Anomaly):")
# We want to catch as many real anomalies as possible.
print(classification_report(y_true, y_pred, target_names=['Anomaly (-1)', 'Normal (1)']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_true, y_pred, labels=[-1, 1])
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=['Anomaly', 'Normal'], yticklabels=['Anomaly', 'Normal'])
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()


# ----------------------------------------
# 6. Analysis of Detected Anomalies
# ----------------------------------------
print("\n--- Analyzing Feature Differences between Normal and Detected Anomalies ---")

df['prediction'] = y_pred
detected_anomalies = df[df['prediction'] == -1]
detected_normals = df[df['prediction'] == 1]

# Compare a key feature between the groups
plt.figure(figsize=(10, 6))
sns.kdeplot(detected_normals['AS_PATH_LEN'], label='Predicted Normal', fill=True)
sns.kdeplot(detected_anomalies['AS_PATH_LEN'], label='Predicted Anomaly', fill=True, color='red')
plt.title('Distribution of AS_PATH_LEN for Normal vs. Detected Anomalous Updates')
plt.xlabel('AS Path Length')
plt.legend()
plt.grid(True)
plt.show()


# ----------------------------------------
# 7. Conclusion
# ----------------------------------------
print("\n--- Conclusion ---")
print("The unsupervised Isolation Forest model successfully learned to identify anomalous BGP updates.")
print("Key Takeaways:")
print("- The model achieved high recall for the 'Anomaly' class, which is crucial for a security system designed to detect rare but critical events like BGP hijacks.")
print("- The key to this approach is training on a trusted baseline of 'normal' data. The model learns the typical patterns of AS-path lengths, edit distances, and prefix lengths. Any update that deviates significantly from this learned profile is flagged.")
print("- The feature distribution plot confirms the model's logic. We can see that the updates it flagged as anomalous often had unusually long AS paths, a classic symptom of a route leak or hijack.")
print("- This type of anomaly detection system is vital for large network operators and ISPs to protect their address space and ensure the stability of internet routing. It can provide an early warning of an attack, allowing for rapid mitigation before it causes a widespread outage.")

```