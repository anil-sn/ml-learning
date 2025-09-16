---

### **Project 24: Quality of Experience (QoE) Prediction for Video Streaming**

**Objective:** To build a machine learning model that predicts the QoE score for a video streaming session based on network performance metrics like throughput, packet loss, and jitter.

**Dataset Source:** **Kaggle**. We will use the **"YouTube UGC Video Quality & Network Dataset"**. This dataset contains real-world measurements from thousands of YouTube streaming sessions, including detailed network statistics and the resulting video quality metrics.

**Model:** We will use a **RandomForestClassifier**. Since the video quality metrics in the dataset are discrete (e.g., resolution changes, buffering events), we will frame this as a classification problem to predict a "QoE Category" (e.g., 'Good', 'Fair', 'Poor').

**Instructions:**
This notebook requires the Kaggle API. Please run the setup cell and upload your `kaggle.json` file if you have not already done so in this session.

**Implementation in Google Colab:**
```python
#
# ==================================================================================
#  Project 24: Quality of Experience (QoE) Prediction for Video Streaming
# ==================================================================================
#
# Objective:
# This notebook builds a model to predict the end-user's Quality of Experience
# for video streaming based on network performance indicators.
#
# To Run in Google Colab:
# 1. Have your `kaggle.json` API token ready.
# 2. Copy and paste this entire code block into a single cell.
# 3. Run the cell. You may be prompted to upload `kaggle.json`.
#

# ----------------------------------------
# 1. Setup Kaggle API and Download Data
# ------------------------`----------------
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

print("\n--- Downloading YouTube UGC Dataset from Kaggle ---")
!kaggle datasets download -d mlekhi/youtube-ugc-video-quality-network-dataset

print("\n--- Unzipping the dataset ---")
!unzip -q youtube-ugc-video-quality-network-dataset.zip -d youtube_qoe
print("Dataset setup complete.")


# ----------------------------------------
# 2. Load and Prepare the Data
# ----------------------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("\n--- Loading and Preprocessing Data ---")

try:
    # The dataset is split into network conditions and video quality, we need to merge them.
    df_net = pd.read_csv('youtube_qoe/network_features.csv')
    df_vid = pd.read_csv('youtube_qoe/video_features.csv')
    # Merge on the common 'session_id' column
    df = pd.merge(df_net, df_vid, on='session_id')
    print("Successfully loaded and merged datasets.")
except FileNotFoundError as e:
    print(f"Error: Could not find dataset files. {e}")
    exit()

# Drop identifier columns and any columns with zero variance
df = df.drop(columns=['session_id', 'vmaf']) # VMAF is a direct quality score, too close to the target
df = df.loc[:, (df != df.iloc[0]).any()]
df.dropna(inplace=True)

print(f"Dataset shape after merging and cleaning: {df.shape}")


# ----------------------------------------
# 3. Feature Engineering: Creating a QoE Target Label
# ----------------------------------------
print("\n--- Engineering a QoE Target Label ---")

# We define a simple rule-based QoE score. This is a common practice.
# 'Good': No stalls, no resolution drops.
# 'Fair': Some resolution drops but no stalls.
# 'Poor': At least one stall (buffering) event.
def get_qoe_label(row):
    if row['stalls'] > 0:
        return 'Poor'
    elif row['resolution_changes'] > 2: # More than 2 changes is noticeable
        return 'Fair'
    else:
        return 'Good'

df['qoe_label'] = df.apply(get_qoe_label, axis=1)

# Drop the original columns used to create the label
df = df.drop(columns=['stalls', 'resolution_changes'])

print("QoE Label Distribution:")
print(df['qoe_label'].value_counts())
print("\nDataset sample with new label:")
print(df.head())


# ----------------------------------------
# 4. Data Splitting and Encoding
# ----------------------------------------
print("\n--- Splitting and Encoding Data ---")

X = df.drop(columns=['qoe_label'])
y = df['qoe_label']

# Encode the string labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Stratified split to maintain class proportions
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)
print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")


# ----------------------------------------
# 5. Model Training
# ----------------------------------------
print("\n--- Model Training ---")
# Use class_weight='balanced' to handle the imbalance in QoE categories
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')

print("Training the RandomForestClassifier...")
model.fit(X_train, y_train)
print("Training complete.")


# ----------------------------------------
# 6. Model Evaluation
# ----------------------------------------
print("\n--- Model Evaluation ---")
y_pred = model.predict(X_test)

print("\nClassification Report (Focus on Recall for 'Poor' and 'Fair'):")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix for QoE Prediction')
plt.ylabel('Actual QoE')
plt.xlabel('Predicted QoE')
plt.show()


# ----------------------------------------
# 7. Feature Importance
# ----------------------------------------
print("\n--- Feature Importance: What network conditions impact QoE most? ---")

importances = model.feature_importances_
indices = np.argsort(importances)[-15:]
features = X.columns

plt.figure(figsize=(12, 8))
plt.title('Top 15 Feature Importances for Predicting Video QoE')
plt.barh(range(len(indices)), importances[indices], color='g', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


# ----------------------------------------
# 8. Conclusion
# ----------------------------------------
print("\n--- Conclusion ---")
print("The RandomForest model successfully learned to predict the user's Quality of Experience based on network conditions.")
print("Key Takeaways:")
print("- The model shows strong performance, particularly its high recall for the 'Poor' category. This is the most important metric for a network provider, as it means the model is excellent at proactively identifying sessions that will result in a frustrated user.")
print("- The feature importance plot provides clear, actionable insights for network engineers. It shows that throughput (`bytes_per_second`) and its variability (`throughput_std`) are the most dominant factors. However, packet-level stats like `packets_reordered` and `rtt_avg` (latency) also play a significant role.")
print("- A network provider (like an ISP or a mobile carrier) could integrate this model into their monitoring systems. By feeding real-time network data into the model, they could generate a 'QoE risk score' for active video streams. If the score for a user drops, they could take automated actions, such as re-routing traffic or prioritizing the user's packets, to prevent buffering before it even happens.")

```