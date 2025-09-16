---

### **Project 5: Predicting Network Device Failure/Degradation**

**Objective:** To build a model that can predict an imminent device failure based on its operational metrics. This allows engineers to perform proactive maintenance, preventing outages.

**Dataset Source:** **Kaggle**. We will use the **"Backblaze Hard Drive Stats"** dataset. Backblaze, a cloud storage provider, publishes daily SMART metric data for the tens of thousands of hard drives in their data centers, including failure information.

**Model:** We will use **LightGBM (Light Gradient Boosting Machine)**. It is a state-of-the-art gradient boosting framework that is extremely fast, memory-efficient, and highly effective for imbalanced tabular data like this.

**Instructions:**
This notebook requires the Kaggle API. If you have already configured it, you can skip the setup. Otherwise, run the setup cell and upload your `kaggle.json` file.

**Implementation in Google Colab:**
```python
#
# ==================================================================================
#  Project 5: Predicting Network Device Failure/Degradation
# ==================================================================================
#
# Objective:
# This notebook builds a predictive maintenance model using hard drive sensor data
# as a proxy for network device metrics. The goal is to predict device failure
# from a highly imbalanced dataset.
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

print("\n--- Downloading Backblaze Hard Drive Dataset from Kaggle ---")
# The full dataset is huge. We will download a specific, manageable portion.
!kaggle datasets download -d anasofiauzsoy/hard-drive-failure-prediction -f 'hdd_latest_select_features.csv'

print("\n--- Unzipping the dataset ---")
!unzip -q hdd_latest_select_features.csv.zip -d .
print("Dataset setup complete.")


# ----------------------------------------
# 2. Load and Prepare the Data
# ----------------------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

print("\n--- Loading and Preprocessing Data ---")

try:
    df = pd.read_csv('hdd_latest_select_features.csv')
except FileNotFoundError:
    print("Error: CSV file not found. Download may have failed.")
    exit()

print(f"Dataset loaded. Shape: {df.shape}")

# Drop non-feature columns
df = df.drop(columns=['date', 'serial_number', 'model'])

# Handle missing values - a common issue with sensor data. We'll use a simple fill.
df.fillna(0, inplace=True)
print("Filled missing values with 0.")

# --- Understand the Class Imbalance ---
# This is the MOST important step for this problem.
failure_counts = df['failure'].value_counts()
failure_rate = failure_counts[1] / (failure_counts[0] + failure_counts[1])
print("\nClass Distribution:")
print(failure_counts)
print(f"Failure Rate: {failure_rate:.4%}")
if failure_rate > 0.1:
    print("Warning: The 'failure' rate is higher than expected. Check the data.")


# ----------------------------------------
# 3. Data Splitting
# ----------------------------------------
print("\n--- Splitting Data for Training and Testing ---")

X = df.drop(columns=['failure'])
y = df['failure']

# Use stratify=y to ensure the tiny proportion of failures is present in both train and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"X_train shape: {X_train.shape}")
print(f"Test set failure distribution:\n{y_test.value_counts()}")


# ----------------------------------------
# 4. Model Training with LightGBM
# ----------------------------------------
print("\n--- Model Training ---")

# To handle the extreme class imbalance, we calculate a weight for the positive class.
# This tells the model to pay much more attention to failure events during training.
scale_pos_weight = failure_counts[0] / failure_counts[1]
print(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")

# Initialize the LightGBM Classifier
model = lgb.LGBMClassifier(
    objective='binary',
    random_state=42,
    n_jobs=-1,
    scale_pos_weight=scale_pos_weight  # Crucial parameter for imbalance
)

print("Training the LightGBM model...")
model.fit(X_train, y_train)
print("Training complete.")


# ----------------------------------------
# 5. Model Evaluation for Imbalanced Data
# ----------------------------------------
print("\n--- Model Evaluation ---")

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1] # Probabilities for the 'failure' class

# --- DO NOT rely on accuracy! ---
# A model predicting 'no failure' all the time would have >99% accuracy.

# The Classification Report is key. We care most about 'recall' for class 1.
print("\nClassification Report (Focus on Recall for class 1):")
print(classification_report(y_test, y_pred, target_names=['No Failure (0)', 'Failure (1)']))

# The Confusion Matrix shows us the actual number of correct/incorrect predictions.
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Failure', 'Failure'], yticklabels=['No Failure', 'Failure'])
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()
print("TN (Top-Left): Correctly predicted no failure.")
print("FP (Top-Right): Incorrectly predicted failure (False Alarm).")
print("FN (Bottom-Left): MISSED a real failure (Most Critical Error!).")
print("TP (Bottom-Right): Correctly predicted failure (True Positive).")


# --- Precision-Recall Curve ---
# This is a much better visualization than ROC/AUC for imbalanced datasets.
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall, precision)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='best')
plt.grid(True)
plt.show()


# ----------------------------------------
# 6. Feature Importance
# ----------------------------------------
print("\n--- Feature Importance ---")
lgb.plot_importance(model, max_num_features=15, height=0.8, figsize=(10, 8))
plt.title('Top 15 Feature Importances for Predicting Failure')
plt.show()


# ----------------------------------------
# 7. Conclusion
# ----------------------------------------
print("\n--- Conclusion ---")
print("This notebook demonstrates a practical approach to predictive maintenance, a problem defined by severe class imbalance.")
print("Key Takeaways:")
print(f"- The model's recall for the 'Failure' class is the most important metric. It tells us the percentage of actual failures we successfully caught. A high recall is critical to prevent outages.")
print("- The Confusion Matrix showed us the exact number of missed failures (False Negatives), which is the primary business risk.")
print("- The Precision-Recall curve provides a clear view of the trade-off between raising false alarms (low precision) and catching more failures (high recall).")
print("- The feature importance plot shows which metrics (e.g., 'smart_5_raw', 'smart_187_raw') are the strongest predictors, guiding what network engineers should monitor most closely.")
```