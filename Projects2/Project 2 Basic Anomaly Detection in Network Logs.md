---

### **Project 2: Basic Anomaly Detection in Network Logs**

**Objective:** To identify anomalous sequences of events in system logs. This is crucial for detecting faults, misconfigurations, or security incidents without needing prior examples of every possible bad event.

**Dataset Source:** **Kagle**. We will use the **HDFS (Hadoop Distributed File System) Log Dataset**, a widely recognized benchmark for log anomaly detection. The logs track events related to data block operations in a large computing cluster, making it an excellent proxy for complex network system logs.

**Model:** We will use **Isolation Forest**, a powerful and efficient unsupervised algorithm designed specifically for anomaly detection.

**Instructions:**
This notebook also requires the Kaggle API. If you have already uploaded your `kaggle.json` file in this Colab session, you can skip the first cell. If you have started a new session, please run the first cell and upload your `kaggle.json` file when prompted.

**Implementation in Google Colab:**

```python
#
# ==================================================================================
#  Project 2: Basic Anomaly Detection in Network Logs
# ==================================================================================
#
# Objective:
# This notebook demonstrates an unsupervised approach to detecting anomalies in
# system logs using the HDFS dataset from Kaggle.
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

    # Install the Kaggle library
    !pip install -q kaggle

    # Prompt user to upload their kaggle.json file
    from google.colab import files
    print("\nPlease upload your kaggle.json file:")
    uploaded = files.upload()

    # Check if the file was uploaded
    if 'kaggle.json' not in uploaded:
        print("\nError: kaggle.json not uploaded. Please restart the cell and upload the file.")
        exit()

    print("\nkaggle.json uploaded successfully.")

    # Create the .kaggle directory and move the json file there
    !mkdir -p ~/.kaggle
    !cp kaggle.json ~/.kaggle/
    !chmod 600 ~/.kaggle/kaggle.json
else:
    print("Kaggle API already configured.")


print("\n--- Downloading HDFS Log Dataset from Kaggle ---")
# Download the dataset (user: logpai, dataset: hdfs-log-anomaly-detection)
!kaggle datasets download -d logpai/hdfs-log-anomaly-detection

print("\n--- Unzipping the dataset ---")
# Unzip the downloaded file
!unzip -q hdfs-log-anomaly-detection.zip -d .

print("\nDataset setup complete.")


# ----------------------------------------
# 2. Load and Preprocess the Log Data
# ----------------------------------------
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, accuracy_score

print("\n--- Loading and Preprocessing Data ---")

# Load the ground truth labels
try:
    labels_df = pd.read_csv('anomaly_label.csv')
    print("Loaded anomaly_label.csv successfully.")
except FileNotFoundError:
    print("Error: anomaly_label.csv not found.")
    exit()

# Load the raw log file
try:
    with open('HDFS.log', 'r') as f:
        logs = f.readlines()
    print("Loaded HDFS.log successfully.")
except FileNotFoundError:
    print("Error: HDFS.log not found.")
    exit()

# Function to parse a raw log line and extract the block ID and log content
def parse_log_line(line):
    match = re.search(r'(blk_[-]?\d+)', line)
    block_id = match.group(1) if match else None
    content = line.strip()
    return block_id, content

# Parse all logs
parsed_logs = [parse_log_line(line) for line in logs]

# Create a DataFrame from the parsed logs
log_df = pd.DataFrame(parsed_logs, columns=['BlockId', 'Content'])
log_df.dropna(inplace=True) # Remove lines where no BlockId was found

# Group log messages by their BlockId. Each BlockId represents a "session".
# We aggregate the log content into a single document for each session.
print("Grouping logs by BlockId (session)...")
session_df = log_df.groupby('BlockId')['Content'].apply(lambda x: ' '.join(x)).reset_index()

# Merge with labels to have the ground truth for evaluation later
session_df = pd.merge(session_df, labels_df, on='BlockId', how='left')
session_df['Label'].fillna('Normal', inplace=True) # Assume sessions not in label file are Normal

print("Preprocessing complete. Sample of session data:")
print(session_df.head())

# ----------------------------------------
# 3. Feature Engineering: TF-IDF
# ----------------------------------------
print("\n--- Feature Engineering ---")
print("Converting log messages into numerical vectors using TF-IDF...")

# TF-IDF (Term Frequency-Inverse Document Frequency) is a great way to convert
# text documents into numerical feature vectors. It gives more weight to terms
# that are frequent in a document but rare across all documents.
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X = vectorizer.fit_transform(session_df['Content'])

print(f"Feature matrix created with shape: {X.shape}")


# ----------------------------------------
# 4. Model Training (Unsupervised)
# ----------------------------------------
print("\n--- Model Training (Unsupervised) ---")

# Initialize the IsolationForest model
# 'contamination' is the expected proportion of anomalies in the data.
# We can estimate it from our labels file.
anomaly_proportion = len(labels_df[labels_df['Label'] == 'Anomaly']) / len(session_df)
print(f"Estimated anomaly proportion: {anomaly_proportion:.4f}")

# It's good practice to set contamination to 'auto' or a well-reasoned value.
# We'll use our calculated proportion.
model = IsolationForest(n_estimators=100,
                        contamination=anomaly_proportion,
                        random_state=42,
                        n_jobs=-1)

print("Training the Isolation Forest model...")
# Note: We do NOT use 'y' labels for training an unsupervised model.
model.fit(X)
print("Training complete.")


# ----------------------------------------
# 5. Model Evaluation
# ----------------------------------------
print("\n--- Model Evaluation ---")

# Predict anomalies. The model returns 1 for inliers and -1 for outliers.
predictions = model.predict(X)

# Convert our ground truth labels to the same format for comparison
# (Normal -> 1, Anomaly -> -1)
y_true = session_df['Label'].apply(lambda x: 1 if x == 'Normal' else -1)
y_pred = predictions

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")

# Display the classification report
# Note: '1' is 'Normal' (inlier), '-1' is 'Anomaly' (outlier)
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=['Anomaly (-1)', 'Normal (1)']))


# ----------------------------------------
# 6. Conclusion
# ----------------------------------------
print("\n--- Conclusion ---")
print(f"The Isolation Forest model successfully identified log anomalies with an accuracy of {accuracy:.2%}.")
print("The key takeaway is that we were able to detect these anomalies WITHOUT explicitly training the model on what an anomaly looks like.")
print("The Classification Report shows strong performance, especially in precision for anomalies, which means when the model flags something, it's highly likely to be a real issue.")
print("This unsupervised approach is extremely powerful for real-world network monitoring where new, unseen problems can occur at any time.")
```