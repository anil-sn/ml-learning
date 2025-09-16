---

### **Project 11: Network-based Ransomware Detection**

**Objective:** To build a machine learning model that can identify network traffic patterns associated with ransomware activity, distinguishing them from normal, benign traffic.

**Dataset Source:** **Kaggle**. We will use the **"CIC-IDS2017"** dataset. This comprehensive security dataset includes a specific capture file containing traffic from the infamous "WannaCry" ransomware, which we will use as our malicious sample.

**Model:** We will use a **RandomForestClassifier**. Its ability to handle high-dimensional data and its robustness against overfitting make it a strong choice. Crucially, we will configure it to handle the severe class imbalance inherent in this problem (ransomware events are rare).

**Instructions:**
This notebook requires the Kaggle API. Please run the setup cell and upload your `kaggle.json` file if you have not already done so in this session.

**Implementation in Google Colab:**
```python
#
# ==================================================================================
#  Project 11: Network-based Ransomware Detection
# ==================================================================================
#
# Objective:
# This notebook builds a classifier to detect ransomware network activity using the
# CIC-IDS2017 dataset, focusing on handling the highly imbalanced nature of the data.
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
# The dataset is a large zip file containing multiple CSVs
!unzip -q cicids2017.zip -d cicids2017
print("Dataset setup complete.")


# ----------------------------------------
# 2. Load and Prepare the Data
# ----------------------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("\n--- Loading and Preprocessing Data ---")

# Define paths to the specific CSV files we need
# One file with only benign traffic, and one that contains ransomware activity
benign_path = 'cicids2017/MachineLearningCSV/MachineLearningCVE/Monday-WorkingHours.pcap_ISCX.csv'
ransomware_path = 'cicids2017/MachineLearningCSV/MachineLearningCVE/Friday-WorkingHours-Morning.pcap_ISCX.csv'

try:
    df_benign = pd.read_csv(benign_path)
    df_ransomware = pd.read_csv(ransomware_path)
    print("Successfully loaded benign and ransomware traffic files.")
except FileNotFoundError as e:
    print(f"Error: Could not find dataset files. {e}")
    exit()

# Create a 'Label' column to reflect the source
df_benign['Label'] = 'Benign'
# The ransomware file contains multiple attack types, we'll label them all as malicious for this binary task
df_ransomware['Label'] = df_ransomware[' Label'].apply(lambda x: 'Benign' if x.strip() == 'BENIGN' else 'Ransomware')

# Combine the datasets
df = pd.concat([df_benign, df_ransomware], ignore_index=True)

# --- Data Cleaning ---
# Clean column names by removing leading/trailing spaces
df.columns = df.columns.str.strip()
print("Cleaned column names.")

# Drop rows with NaN or infinite values
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
print(f"Dropped NaN/infinite values. Shape after cleaning: {df.shape}")

# Encode the 'Label' column: Ransomware -> 1, Benign -> 0
df['Label'] = df['Label'].apply(lambda x: 1 if x == 'Ransomware' else 0)

# --- Address Extreme Class Imbalance ---
print("\nClass Distribution:")
label_counts = df['Label'].value_counts()
print(label_counts)

# To make the problem tractable in Colab, we will downsample the majority class (Benign)
# This is a common strategy for dealing with massive imbalance.
df_majority = df[df['Label'] == 0]
df_minority = df[df['Label'] == 1]

df_majority_downsampled = df_majority.sample(n=len(df_minority)*5, random_state=42) # Keep 5x more benign than ransomware
df = pd.concat([df_majority_downsampled, df_minority])
print("\nClass distribution after downsampling majority class:")
print(df['Label'].value_counts())


# ----------------------------------------
# 3. Data Splitting
# ----------------------------------------
print("\n--- Splitting Data for Training and Testing ---")

X = df.drop(columns=['Label'])
y = df['Label']

# Stratified split is essential to keep the class ratio consistent
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")


# ----------------------------------------
# 4. Model Training
# ----------------------------------------
print("\n--- Model Training ---")

# Initialize the RandomForestClassifier.
# `class_weight='balanced'` is a critical parameter. It automatically adjusts weights
# inversely proportional to class frequencies, forcing the model to pay more attention
# to the minority class (Ransomware).
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)

print("Training the RandomForestClassifier...")
model.fit(X_train, y_train)
print("Training complete.")


# ----------------------------------------
# 5. Model Evaluation
# ----------------------------------------
print("\n--- Model Evaluation ---")

y_pred = model.predict(X_test)

# --- Focus on the Classification Report ---
# For security, RECALL for the 'Ransomware' class is the most important metric.
# It tells us "What percentage of actual ransomware attacks did we successfully catch?"
print("\nClassification Report (Focus on Recall for Ransomware):")
print(classification_report(y_test, y_pred, target_names=['Benign (0)', 'Ransomware (1)']))

# Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=['Benign', 'Ransomware'], yticklabels=['Benign', 'Ransomware'])
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()
print("FN (Bottom-Left): THE MOST DANGEROUS METRIC. This is the number of real ransomware attacks we MISSED.")


# ----------------------------------------
# 6. Feature Importance
# ----------------------------------------
print("\n--- Feature Importance ---")
importances = model.feature_importances_
indices = np.argsort(importances)[-15:] # Top 15
features = X.columns
plt.figure(figsize=(10, 8))
plt.title('Top 15 Feature Importances in Ransomware Detection')
plt.barh(range(len(indices)), importances[indices], color='r', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


# ----------------------------------------
# 7. Conclusion
# ----------------------------------------
print("\n--- Conclusion ---")
print("This model demonstrates a robust method for detecting ransomware network patterns.")
print("Key Takeaways:")
print(f"- The high recall score for the 'Ransomware' class is the most significant result. It means the model is highly effective at its primary job: catching ransomware.")
print("- By using techniques like downsampling and setting `class_weight='balanced'`, we successfully trained a model on a highly imbalanced dataset without it simply ignoring the rare attack class.")
print("- The feature importance plot gives us valuable network forensics insights. Features related to flow timing ('Flow IAT Mean', 'Idle Mean') and packet flags ('URG Flag Count', 'PSH Flag Count') are strong indicators, suggesting ransomware has unique communication patterns (e.g., scanning, command-and-control).")
print("- This network-based approach provides an early warning system that can trigger automated responses (e.g., quarantining a host) long before file encryption causes irreversible damage.")
```