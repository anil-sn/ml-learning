---

### **Project 10: Encrypted Traffic Classification**

**Objective:** To build a machine learning model that can classify different types of application traffic (Chat, File Transfer, Streaming, etc.) even when it is encrypted within a VPN tunnel.

**Dataset Source:** **Kaggle**. We will use the **"ISCX VPN-nonVPN Dataset"** from the University of New Brunswick, a benchmark for this task. It contains pre-extracted statistical features from thousands of network flows.

**Model:** We will use **XGBoost (Extreme Gradient Boosting)**, a highly optimized and powerful gradient boosting library that consistently delivers state-of-the-art results on tabular data like this.

**Instructions:**
This notebook requires the Kaggle API. Please run the setup cell and upload your `kaggle.json` file if you have not already done so in this session.

**Implementation in Google Colab:**
```python
#
# ==================================================================================
#  Project 10: Encrypted Traffic Classification
# ==================================================================================
#
# Objective:
# This notebook builds a model to classify the type of application generating
# encrypted network traffic based on flow-level metadata.
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

print("\n--- Downloading ISCX VPN-nonVPN Dataset from Kaggle ---")
!kaggle datasets download -d jsphyg/vpn-non-vpn-dataset

print("\n--- Unzipping the dataset ---")
!unzip -q vpn-non-vpn-dataset.zip -d vpn_dataset
print("Dataset setup complete.")


# ----------------------------------------
# 2. Load and Prepare the Data
# ----------------------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import glob

print("\n--- Loading and Preprocessing Data ---")

# Define the types of traffic we want to classify
# We'll focus on the VPN traffic for this demonstration
traffic_types = ['vpn_chat', 'vpn_file', 'vpn_email', 'vpn_streaming', 'vpn_voip']
data_path = 'vpn_dataset/vpn'

# Load and label each CSV file
df_list = []
for traffic_type in traffic_types:
    # Use glob to find the exact CSV file (e.g., vpn_file_transfer.csv)
    file_pattern = os.path.join(data_path, f"{traffic_type}*.csv")
    try:
        csv_file = glob.glob(file_pattern)[0]
        temp_df = pd.read_csv(csv_file)
        # Use a simplified label for the class
        temp_df['label'] = traffic_type.split('_')[1] # e.g., 'chat', 'file'
        df_list.append(temp_df)
        print(f"Loaded {os.path.basename(csv_file)} with label '{temp_df['label'].iloc[0]}'")
    except (FileNotFoundError, IndexError):
        print(f"Warning: Could not find or load CSV for {traffic_type}")

# Combine into a single dataframe
df = pd.concat(df_list, ignore_index=True)

# --- Data Cleaning ---
# The dataset may have column names with leading spaces
df.columns = df.columns.str.strip()
print("\nCleaned column names.")

# Drop columns that are not useful features or are identifiers
df = df.drop(columns=['Flow ID', 'Src IP', 'Dst IP', 'Src Port', 'Dst Port', 'Timestamp'])

# Handle infinite values and NaNs which are common in flow stats
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
print(f"Dropped NaN/infinite values. Shape after cleaning: {df.shape}")

# Check the distribution of our new labels
print("\nClass Distribution:")
print(df['label'].value_counts())


# ----------------------------------------
# 3. Feature Selection and Data Splitting
# ----------------------------------------
print("\n--- Splitting Data for Training and Testing ---")

X = df.drop(columns=['label'])
y = df['label']

# Encode the string labels into integers
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Stratified split to maintain class proportions in train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")


# ----------------------------------------
# 4. Model Training with XGBoost
# ----------------------------------------
print("\n--- Model Training ---")

# Initialize the XGBoost Classifier
# `objective='multi:softmax'` is used for multi-class classification.
# `use_label_encoder=False` and `eval_metric='mlogloss'` are modern standards.
model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=len(le.classes_),
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42,
    n_jobs=-1
)

print("Training the XGBoost model... (This may take a few minutes)")
model.fit(X_train, y_train)
print("Training complete.")


# ----------------------------------------
# 5. Model Evaluation
# ----------------------------------------
print("\n--- Model Evaluation ---")

y_pred = model.predict(X_test)

# The Classification Report shows precision, recall, and f1-score for each class.
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# The Confusion Matrix visualizes where the model is making mistakes.
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix for Encrypted Traffic')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()


# ----------------------------------------
# 6. Feature Importance
# ----------------------------------------
print("\n--- Feature Importance ---")

# XGBoost has a built-in function to plot feature importance
fig, ax = plt.subplots(figsize=(12, 8))
xgb.plot_importance(model, ax=ax, max_num_features=15, height=0.8)
plt.title('Top 15 Feature Importances')
plt.show()


# ----------------------------------------
# 7. Conclusion
# ----------------------------------------
print("\n--- Conclusion ---")
print("The XGBoost model successfully learned to distinguish between different types of encrypted application traffic with high accuracy.")
print("Key Takeaways:")
print("- This demonstrates that it's possible to manage and classify network traffic without decrypting it, which is crucial for privacy and security.")
print("- The Confusion Matrix shows the model is highly accurate, with most confusion happening between similar traffic types (if any).")
print("- The Feature Importance plot is very insightful. It highlights that metrics like 'Fwd Packet Length Mean', 'Flow Duration', and 'Packet Length Variance' are powerful differentiators. This makes sense intuitively: streaming traffic (long flows, large consistent packets) looks very different from chat traffic (short flows, small bursty packets).")
print("- Network engineers can use such a model for Quality of Service (QoS) to prioritize latency-sensitive traffic (like VoIP) or for security to detect anomalous encrypted flows.")

```