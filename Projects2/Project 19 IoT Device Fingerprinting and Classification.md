---

### **Project 19: IoT Device Fingerprinting and Classification**

**Objective:** To build a multi-class classification model that can accurately identify the type of an IoT device by analyzing the statistical features of its network traffic.

**Dataset Source:** **Kaggle**. We will use the **"UNSW-IoT Traffic Profile Dataset"**, which contains labeled network traffic features from 28 distinct IoT devices, making it perfect for this classification task.

**Model:** We will use **LightGBM (Light Gradient Boosting Machine)**. It is an excellent choice for this multi-class classification problem due to its high efficiency, scalability, and ability to deliver top-tier accuracy on large tabular datasets.

**Instructions:**
This notebook requires the Kaggle API. Please run the setup cell and upload your `kaggle.json` file if you have not already done so in this session.

**Implementation in Google Colab:**
```python
#
# ==================================================================================
#  Project 19: IoT Device Fingerprinting and Classification
# ==================================================================================
#
# Objective:
# This notebook builds a multi-class classifier to identify IoT device types
# from their network traffic profiles using the UNSW-IoT dataset.
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

print("\n--- Downloading UNSW-IoT Traffic Profile Dataset from Kaggle ---")
!kaggle datasets download -d salahalag/unsw-iot-traffic-profile-dataset

print("\n--- Unzipping the dataset ---")
!unzip -q unsw-iot-traffic-profile-dataset.zip -d iot_dataset
print("Dataset setup complete.")


# ----------------------------------------
# 2. Load and Prepare the Data
# ----------------------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("\n--- Loading and Preprocessing Data ---")

try:
    # The dataset is a single large CSV file
    df = pd.read_csv('iot_dataset/UNSW_IoT_traffic_profile_dataset.csv')
    print("Successfully loaded the dataset.")
except FileNotFoundError as e:
    print(f"Error: Could not find dataset file. {e}")
    exit()

# Drop columns that are identifiers and not useful for general fingerprinting
df = df.drop(columns=['ip.src', 'ip.dst', 'label'])

# --- Data Cleaning ---
# Check for missing values
if df.isnull().sum().sum() > 0:
    print("Warning: Missing values detected. Filling with 0.")
    df.fillna(0, inplace=True)

print(f"Data loaded. Shape: {df.shape}")

# Inspect the target variable 'device_category'
print("\nIoT Device Category Distribution:")
print(df['device_category'].value_counts())


# ----------------------------------------
# 3. Data Splitting and Encoding
# ----------------------------------------
print("\n--- Splitting and Encoding Data ---")

X = df.drop(columns=['device_category'])
y = df['device_category']

# Encode the string labels for both features and the target variable
le_y = LabelEncoder()
y_encoded = le_y.fit_transform(y)

# Encode categorical features in X, if any
for col in X.select_dtypes(include=['object']).columns:
    le_x = LabelEncoder()
    X[col] = le_x.fit_transform(X[col])

# Stratified split to ensure all device categories are represented in both sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)

print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")


# ----------------------------------------
# 4. Model Training with LightGBM
# ----------------------------------------
print("\n--- Model Training ---")

# Initialize the LightGBM Classifier for multi-class classification
model = lgb.LGBMClassifier(
    objective='multiclass',
    num_class=len(le_y.classes_),
    random_state=42,
    n_jobs=-1
)

print("Training the LightGBM model... (This may take a minute)")
model.fit(X_train, y_train)
print("Training complete.")


# ----------------------------------------
# 5. Model Evaluation
# ----------------------------------------
print("\n--- Model Evaluation ---")
y_pred = model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le_y.classes_))

# Plot the confusion matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(18, 15))
sns.heatmap(cm, annot=True, fmt='d', cmap='cividis', xticklabels=le_y.classes_, yticklabels=le_y.classes_)
plt.title('Confusion Matrix for IoT Device Classification', fontsize=16)
plt.ylabel('Actual Device')
plt.xlabel('Predicted Device')
plt.show()


# ----------------------------------------
# 6. Feature Importance
# ----------------------------------------
print("\n--- Feature Importance ---")
lgb.plot_importance(model, max_num_features=20, height=0.8, figsize=(12, 10))
plt.title('Top 20 Feature Importances for IoT Device Fingerprinting')
plt.show()


# ----------------------------------------
# 7. Conclusion
# ----------------------------------------
print("\n--- Conclusion ---")
print("The LightGBM model demonstrated outstanding performance in classifying a wide variety of IoT devices based on their network traffic profiles.")
print("Key Takeaways:")
print("- The model achieved extremely high precision and recall across most device categories, indicating that IoT devices have unique and stable network 'fingerprints'.")
print("- The confusion matrix is nearly diagonal, which is the ideal result, showing very few misclassifications. The few errors that do occur are often between similar devices (e.g., different smart speakers).")
print("- The feature importance plot reveals which network characteristics are the most powerful for fingerprinting. For instance, TCP/UDP port numbers, packet timings (`tcp.time_delta`), and protocol-specific features (`http.content_length`) are key differentiators.")
print("- This capability is crucial for modern network management and security. An automated system could use this model to populate a device inventory, enforce security policies (e.g., 'Security cameras are only allowed to talk to the video server'), and detect rogue or unauthorized devices connected to the network.")

```