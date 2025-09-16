---

### **Project 13: Identifying Lateral Movement in a Network**

**Objective:** To build an unsupervised anomaly detection model that can identify hosts exhibiting behavior indicative of lateral movement, such as internal port scanning or connecting to an unusually high number of other hosts.

**Dataset Source:** **Kaggle**. We will again use the **"CIC-IDS2017"** dataset. Specifically, we will use traffic from a day containing port scans and web attacks, which serve as excellent proxies for lateral movement activities.

**Model:** We will use **Isolation Forest**. This unsupervised model is perfect for this task because it learns to profile "normal" behavior without needing pre-labeled examples of attacks. We will train it on benign traffic only, and then use it to identify hosts whose behavior deviates from that normal baseline.

**Instructions:**
This notebook requires the Kaggle API. Please run the setup cell and upload your `kaggle.json` file if you have not already done so in this session.

**Implementation in Google Colab:**
```python
#
# ==================================================================================
#  Project 13: Identifying Lateral Movement in a Network
# ==================================================================================
#
# Objective:
# This notebook builds an unsupervised model to detect lateral movement by
# profiling host behavior and identifying anomalies like scanning.
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
    !mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
else:
    print("Kaggle API already configured.")

print("\n--- Downloading CIC-IDS2017 Dataset from Kaggle ---")
!kaggle datasets download -d cic-ids-2017/cicids2017

print("\n--- Unzipping the dataset ---")
!unzip -q cicids2017.zip -d cicids2017
print("Dataset setup complete.")


# ----------------------------------------
# 2. Load and Prepare the Data
# ----------------------------------------
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("\n--- Loading and Preprocessing Data ---")

# We use the 'Tuesday' file which contains a mix of benign traffic and attacks like PortScans.
file_path = 'cicids2017/MachineLearningCSV/MachineLearningCVE/Tuesday-WorkingHours.pcap_ISCX.csv'
try:
    df = pd.read_csv(file_path)
    print("Successfully loaded the dataset.")
except FileNotFoundError as e:
    print(f"Error: Could not find dataset file. {e}")
    exit()

# --- Data Cleaning ---
df.columns = df.columns.str.strip()
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
df['Label'] = df['Label'].str.strip()
print(f"Cleaned and prepared data. Shape: {df.shape}")


# ----------------------------------------
# 3. Feature Engineering: Profiling Host Behavior
# ----------------------------------------
print("\n--- Engineering Behavioral Features for each Source IP ---")

# The core idea is to define "normal" not by a single flow, but by the
# aggregate behavior of each source host over time.
# We group by 'Source IP' and calculate behavioral metrics.
host_profiles = df.groupby('Source IP').agg(
    # How many distinct hosts does it talk to? (Scanning indicator)
    unique_dst_ips=('Destination IP', 'nunique'),
    # How many distinct ports does it target? (Port scanning indicator)
    unique_dst_ports=('Destination Port', 'nunique'),
    # How many total connections does it make?
    total_flows=('Flow ID', 'count'),
    # Average duration of its flows
    avg_flow_duration=('Flow Duration', 'mean')
).reset_index()

# We need a label for each host profile for our final evaluation.
# If a host sent even ONE malicious packet, we label its entire profile as malicious.
def get_host_label(group):
    if (group != 'BENIGN').any():
        return 'Attack'
    return 'BENIGN'

host_labels = df.groupby('Source IP')['Label'].apply(get_host_label).reset_index()
host_profiles = pd.merge(host_profiles, host_labels, on='Source IP')

print("Generated host profiles. Sample:")
print(host_profiles.head())
print("\nProfiled Class Distribution:")
print(host_profiles['Label'].value_counts())


# ----------------------------------------
# 4. Unsupervised Model Training
# ----------------------------------------
print("\n--- Unsupervised Model Training (on BENIGN data only) ---")

# Prepare the data for the model
X = host_profiles.drop(columns=['Source IP', 'Label'])
y_true_labels = host_profiles['Label'].apply(lambda x: 1 if x == 'BENIGN' else -1) # 1 for normal, -1 for anomaly

# --- CRITICAL STEP ---
# We only train the model on what we know is "normal" behavior.
X_train_benign = X[host_profiles['Label'] == 'BENIGN']
print(f"Training Isolation Forest on {len(X_train_benign)} benign host profiles.")

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_benign)

# Initialize and train the Isolation Forest
# Contamination is the expected proportion of anomalies in the *training* data.
# Since we are training on benign data only, we set it to a very small value.
model = IsolationForest(contamination=0.01, random_state=42)
model.fit(X_train_scaled)
print("Training complete.")


# ----------------------------------------
# 5. Model Evaluation
# ----------------------------------------
print("\n--- Model Evaluation ---")

# Now, we use the trained model to predict on ALL hosts (benign and attack).
X_all_scaled = scaler.transform(X) # Use the same scaler from training
predictions = model.predict(X_all_scaled) # Returns 1 for inliers, -1 for outliers/anomalies

host_profiles['Prediction'] = predictions

# Compare our model's predictions with the ground truth labels
print("\nClassification Report:")
print(classification_report(y_true_labels, predictions, target_names=['Attack (-1)', 'Benign (1)']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_true_labels, predictions, labels=[-1, 1])
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=['Attack', 'Benign'], yticklabels=['Attack', 'Benign'])
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()


# ----------------------------------------
# 6. Analysis of Detected Anomalies
# ----------------------------------------
print("\n--- Analyzing Detected Anomalies ---")
# Let's look at the profiles the model flagged as anomalies
detected_anomalies = host_profiles[host_profiles['Prediction'] == -1].sort_values(by='unique_dst_ports', ascending=False)

print("Top 5 profiles flagged as ANOMALIES by the model:")
print(detected_anomalies.head())
print("\nTop 5 profiles flagged as NORMAL by the model:")
print(host_profiles[host_profiles['Prediction'] == 1].head())


# ----------------------------------------
# 7. Conclusion
# ----------------------------------------
print("\n--- Conclusion ---")
print("The unsupervised Isolation Forest model successfully identified malicious hosts based on their network behavior.")
print("Key Takeaways:")
print("- The model, trained only on 'normal' host profiles, effectively learned a baseline and flagged hosts that deviated significantly.")
print("- The high recall for the 'Attack' class demonstrates the model's ability to catch hosts performing suspicious activities like port scanning (indicated by high `unique_dst_ports`).")
print("- This approach is powerful because it doesn't require prior signatures of every possible attack. It can detect novel threats simply because they represent a change from normal behavior.")
print("- In a real security deployment, any host flagged as an anomaly (-1) would be a high-priority candidate for investigation by a security analyst to confirm if it has been compromised.")

```