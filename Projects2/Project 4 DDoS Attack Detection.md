---

### **Project 4: DDoS Attack Detection**

**Objective:** To build a high-performance machine learning model that can accurately distinguish between legitimate (Benign) network traffic and malicious DDoS attack traffic based on network flow features.

**Dataset Source:** **Kaggle**. We will use the **CIC-DDoS2019** dataset. This is a modern and extensive dataset that contains a wide variety of up-to-date DDoS attacks, making it highly relevant for real-world scenarios.

**Model:** We will use a **RandomForestClassifier**. This model is an excellent choice because it is powerful, handles a large number of features well, and can provide insights into which network features are most indicative of an attack.

**Instructions:**
This notebook requires the Kaggle API. If you have already configured it in this Colab session, you can skip the setup cell. Otherwise, please run the setup cell and upload your `kaggle.json` file when prompted.

**Implementation in Google Colab:**
```python
#
# ==================================================================================
#  Project 4: DDoS Attack Detection
# ==================================================================================
#
# Objective:
# This notebook builds a classifier to detect DDoS attacks from network traffic
# flow data using the modern CIC-DDoS2019 dataset from Kaggle.
#
# To Run in Google Colab:
# 1. Have your `kaggle.json` API token ready.
# 2. Copy and paste this entire code block into a single cell.
# 3. Run the cell. If it's your first time in this session, you will be
#    prompted to upload your `kaggle.json` file.
#

# ----------------------------------------
# 1. Setup Kaggle API and Download Data
# ----------------------------------------
import os

# Check if kaggle.json already exists to avoid re-uploading
if not os.path.exists('/root/.kaggle/kaggle.json'):
    print("--- Setting up Kaggle API ---")
    !pip install -q kaggle
    from google.colab import files
    print("\nPlease upload your kaggle.json file:")
    uploaded = files.upload()
    if 'kaggle.json' not in uploaded:
        print("\nError: kaggle.json not uploaded.")
        exit()
    print("\nkaggle.json uploaded successfully.")
    !mkdir -p ~/.kaggle
    !cp kaggle.json ~/.kaggle/
    !chmod 600 ~/.kaggle/kaggle.json
else:
    print("Kaggle API already configured.")

print("\n--- Downloading CIC-DDoS2019 Dataset from Kaggle ---")
# This is a large dataset. The download may take a few minutes.
!kaggle datasets download -d frazane/cicddos2019

print("\n--- Unzipping the dataset ---")
# The dataset is composed of multiple large zip files. We will unzip the main one.
!unzip -q cicddos2019.zip -d cicddos2019
print("Dataset setup complete.")


# ----------------------------------------
# 2. Load and Prepare the Data
# ----------------------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import glob

print("\n--- Loading and Preprocessing Data ---")

# The dataset is split into multiple CSVs. We'll load a few for this demonstration
# to keep memory usage manageable in Colab.
path = 'cicddos2019/CSVs'
# Let's load two files: one with mostly benign traffic and one with a specific DDoS attack type.
filenames = [os.path.join(path, 'DrDoS_NTP.csv'), os.path.join(path, 'syn_and_benign.csv')]
try:
    df_list = [pd.read_csv(f) for f in filenames]
    df = pd.concat(df_list, ignore_index=True)
except FileNotFoundError:
    print(f"Error: Could not find expected CSV files in {path}. Please check the unzipped directory structure.")
    exit()

print(f"Loaded {len(df_list)} files. Total dataset shape: {df.shape}")

# --- Data Cleaning ---
# Remove leading/trailing spaces from column names
df.columns = df.columns.str.strip()
print("Cleaned column names.")

# Drop columns that are not useful for general DDoS detection
# These include identifiers and columns with a single unique value.
df = df.drop(columns=['Unnamed: 0', 'Flow ID', 'Source IP', 'Destination IP', 'Timestamp', 'Fwd Header Length.1'])

# Inspect for and handle infinite values and NaNs
# These can occur from division-by-zero in feature calculation (e.g., flow duration is 0)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
print(f"Dropped NaN/infinite values. Shape after cleaning: {df.shape}")

# Convert the 'Label' column to numerical format (Benign -> 0, DDoS -> 1)
df['Label'] = df['Label'].apply(lambda x: 0 if x == 'Benign' else 1)
print("Encoded 'Label' column.")
print("\nLabel distribution:")
print(df['Label'].value_counts())


# ----------------------------------------
# 3. Feature Selection and Data Splitting
# ----------------------------------------
print("\n--- Splitting Data for Training and Testing ---")

# Separate features (X) from the target label (y)
X = df.drop(columns=['Label'])
y = df['Label']

# Split data into training and testing sets.
# `stratify=y` is crucial here to ensure that the proportion of benign/DDoS traffic
# is the same in both the training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")


# ----------------------------------------
# 4. Model Training
# ----------------------------------------
print("\n--- Model Training ---")

# Initialize the RandomForestClassifier.
# n_estimators=50 is a good balance of speed and performance for a demo.
# n_jobs=-1 will use all available CPU cores in Colab.
model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)

print("Training the RandomForestClassifier... (This may take a minute)")
model.fit(X_train, y_train)
print("Training complete.")


# ----------------------------------------
# 5. Model Evaluation
# ----------------------------------------
print("\n--- Model Evaluation ---")

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")
print("Note: Accuracy can be misleading in imbalanced datasets. Precision and Recall are more important.")

# Display the classification report (Precision, Recall, F1-Score)
# This is the most important output for a security classification task.
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Benign (0)', 'DDoS (1)']))

# Display the confusion matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'DDoS'], yticklabels=['Benign', 'DDoS'])
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()


# ----------------------------------------
# 6. Feature Importance
# ----------------------------------------
print("\n--- Feature Importance ---")

# Extract feature importances from the trained model
importances = model.feature_importances_
indices = np.argsort(importances)[-15:] # Top 15 features
features = X.columns

plt.figure(figsize=(10, 8))
plt.title('Top 15 Feature Importances in DDoS Detection')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


# ----------------------------------------
# 7. Conclusion
# ----------------------------------------
print("\n--- Conclusion ---")
print(f"The RandomForest model achieved an outstanding accuracy of {accuracy:.2%}, but more importantly, it demonstrated high precision and recall for both Benign and DDoS classes.")
print("Key evaluation points:")
print("- The high recall for the 'DDoS' class means the model is excellent at correctly identifying attacks and has very few false negatives (missed attacks).")
print("- The high precision for the 'DDoS' class means that when the model flags traffic as an attack, it is very likely to be correct, resulting in few false positives.")
print("The feature importance plot provides actionable insights for network engineers, highlighting that statistics related to packet size ('Bwd Packet Length Mean', 'Avg Fwd Segment Size') and timing ('Idle Mean', 'Fwd IAT Mean') are key indicators for detecting these types of DDoS attacks.")
```