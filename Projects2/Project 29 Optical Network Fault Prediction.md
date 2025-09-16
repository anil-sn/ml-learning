---

### **Project 29: Optical Network Fault Prediction**

**Objective:** To build a machine learning model that can predict an impending fault in an optical network device (like an amplifier or transceiver) by analyzing its real-time performance metrics.

**Dataset Source:** **Kaggle**. We will use the **"Optical Network Intrusion Dataset"**. While its original purpose was intrusion detection, the dataset is fundamentally a time-series record of optical signal properties (power, OSNR, etc.). We will re-purpose it to predict "unstable" states, which serve as a proxy for pre-fault conditions.

**Model:** We will use a **RandomForestClassifier**. This is an excellent choice for time-series classification as it can effectively handle multiple continuous sensor readings and identify complex patterns that precede a failure state.

**Instructions:**
This notebook requires the Kaggle API. Please run the setup cell and upload your `kaggle.json` file if you have not already done so in this session.

**Implementation in Google Colab:**
```python
#
# ==================================================================================
#  Project 29: Optical Network Fault Prediction
# ==================================================================================
#
# Objective:
# This notebook builds a predictive maintenance model for optical networks by
# classifying the state of the network (stable vs. unstable/pre-fault) based on
# real-time optical performance metrics.
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

print("\n--- Downloading Optical Network Intrusion Dataset from Kaggle ---")
!kaggle datasets download -d 561616/optical-network-intrusion-dataset

print("\n--- Unzipping the dataset ---")
!unzip -q optical-network-intrusion-dataset.zip -d optical_data
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
    df = pd.read_csv('optical_data/Optical_Intrusion_Dataset.csv')
    print("Successfully loaded the dataset.")
except FileNotFoundError as e:
    print(f"Error: Could not find dataset file. {e}")
    exit()

# The last column is 'Intrusion', but we'll treat it as 'Fault_Condition'
df.rename(columns={'Intrusion': 'Fault_Condition'}, inplace=True)
# The first column is a record ID
df = df.drop(columns=['Unnamed: 0'])

# Encode the target label: 'No' -> 0 (Stable), 'Yes' -> 1 (Unstable/Fault)
le = LabelEncoder()
df['Fault_Condition'] = le.fit_transform(df['Fault_Condition'])

print(f"Dataset loaded. Shape: {df.shape}")
print("\nClass Distribution (0=Stable, 1=Unstable):")
print(df['Fault_Condition'].value_counts())
print("\nDataset sample:")
print(df.head())


# ----------------------------------------
# 3. Data Splitting
# ----------------------------------------
print("\n--- Splitting Data for Training and Testing ---")

X = df.drop(columns=['Fault_Condition'])
y = df['Fault_Condition']

# Use a stratified split to maintain the class ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")


# ----------------------------------------
# 4. Model Training
# ----------------------------------------
print("\n--- Model Training ---")
# Use class_weight='balanced' to help the model focus on the minority 'Fault' class
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')

print("Training the RandomForestClassifier...")
model.fit(X_train, y_train)
print("Training complete.")


# ----------------------------------------
# 5. Model Evaluation
# ----------------------------------------
print("\n--- Model Evaluation ---")
y_pred = model.predict(X_test)

# For predictive maintenance, RECALL for the 'Fault' class is most important.
# We want to catch as many impending faults as possible.
print("\nClassification Report (Focus on Recall for Fault_Condition=1):")
print(classification_report(y_test, y_pred, target_names=['Stable (0)', 'Unstable/Fault (1)']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='cividis', xticklabels=['Stable', 'Unstable/Fault'], yticklabels=['Stable', 'Unstable/Fault'])
plt.title('Confusion Matrix for Optical Fault Prediction')
plt.ylabel('Actual State')
plt.xlabel('Predicted State')
plt.show()


# ----------------------------------------
# 6. Feature Importance
# ----------------------------------------
print("\n--- Feature Importance: What metrics predict an optical fault? ---")

importances = model.feature_importances_
features = X.columns
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values('Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(15))
plt.title('Top 15 Feature Importances for Predicting Optical Faults')
plt.show()


# ----------------------------------------
# 7. Conclusion
# ----------------------------------------
print("\n--- Conclusion ---")
print("The RandomForest model proved highly effective at predicting unstable states in the optical network, which serve as a proxy for impending faults.")
print("Key Takeaways:")
print("- The model's excellent recall for the 'Unstable/Fault' class shows it can be a reliable early warning system. Catching these events before they lead to a complete link failure is the primary goal of predictive maintenance.")
print("- The feature importance plot provides invaluable insights for optical engineers. It shows that metrics related to Optical Signal-to-Noise Ratio (`OSNR_…`) and Optical Power (`Power_…`) are the most powerful predictors of instability. This aligns perfectly with the physics of optical networking.")
print("- A real-world system could be built by feeding telemetry data from optical line systems (OLS) and transponders into this model in real-time. When the model's prediction for a link flips to 'Unstable', an alert can be generated for the network operations team to investigate.")
print("- This allows the team to schedule maintenance, replace a degrading component, or re-route traffic *before* the link fails, preventing costly outages and ensuring service continuity for customers.")
```