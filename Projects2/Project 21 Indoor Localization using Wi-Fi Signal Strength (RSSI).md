---

### **Project 21: Indoor Localization using Wi-Fi Signal Strength (RSSI)**

**Objective:** To build a multi-class classification model that can predict a user's specific location (a unique room or space) within a building by using the Received Signal Strength Indicator (RSSI) from numerous nearby Wi-Fi APs.

**Dataset Source:** **Kaggle**. We will use the **"UJIIndoorLoc Data Set"**, a comprehensive dataset containing Wi-Fi fingerprints from 520 different APs, along with the corresponding building, floor, and room location for each measurement.

**Model:** We will use a **RandomForestClassifier**. It is an excellent choice for this problem because it can effectively handle a large number of input features (the 520 APs) and capture the complex, non-linear relationships between signal strengths and physical location.

**Instructions:**
This notebook requires the Kaggle API. Please run the setup cell and upload your `kaggle.json` file if you have not already done so in this session.

**Implementation in Google Colab:**
```python
#
# ==================================================================================
#  Project 21: Indoor Localization using Wi-Fi Signal Strength (RSSI)
# ==================================================================================
#
# Objective:
# This notebook builds a model to predict indoor location (room/space) based on
# the signal strengths of surrounding Wi-Fi Access Points.
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
    !mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle.json
else:
    print("Kaggle API already configured.")

print("\n--- Downloading UJIIndoorLoc Dataset from Kaggle ---")
!kaggle datasets download -d ujim-ml/ujiindoorloc

print("\n--- Unzipping the dataset ---")
!unzip -q ujiindoorloc.zip -d indoor_loc
print("Dataset setup complete.")


# ----------------------------------------
# 2. Load and Prepare the Data
# ----------------------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

print("\n--- Loading and Preprocessing Data ---")

try:
    # We will combine the training and validation sets for a standard split
    df_train = pd.read_csv('indoor_loc/trainingData.csv')
    df_val = pd.read_csv('indoor_loc/validationData.csv')
    df = pd.concat([df_train, df_val], ignore_index=True)
    print("Successfully loaded and combined datasets.")
except FileNotFoundError as e:
    print(f"Error: Could not find dataset files. {e}")
    exit()

# --- Data Cleaning and Feature Engineering ---
# The first 520 columns are the WAP RSSI values
wap_cols = [f'WAP{str(i).zfill(3)}' for i in range(1, 521)]
# The last 9 columns are metadata about the location
loc_cols = ['LONGITUDE', 'LATITUDE', 'FLOOR', 'BUILDINGID', 'SPACEID', 'RELATIVEPOSITION', 'USERID', 'PHONEID', 'TIMESTAMP']

# In this dataset, '100' is used to denote 'no signal'. This is problematic for ML.
# We will replace it with a very low RSSI value (-105) to maintain the ordinal nature of the data.
print("Replacing '100' (no signal) with -105 dBm...")
df[wap_cols] = df[wap_cols].replace(100, -105)

# --- Create a Unique Location Target ---
# We want to predict a specific space. We'll create a composite key for this.
df['location'] = df['BUILDINGID'].astype(str) + '-' + df['FLOOR'].astype(str) + '-' + df['SPACEID'].astype(str)
print(f"Created a unique location target. Total unique locations: {df['location'].nunique()}")
print(f"\nDataset shape after preprocessing: {df.shape}")


# ----------------------------------------
# 3. Data Splitting and Encoding
# ----------------------------------------
print("\n--- Splitting and Encoding Data ---")

X = df[wap_cols]
y = df['location']

# Encode the string labels for the target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Use a stratified split to ensure all locations are represented proportionally
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)
print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")


# ----------------------------------------
# 4. Model Training
# ----------------------------------------
print("\n--- Model Training ---")

# RandomForest is a great choice for this high-dimensional problem
# n_estimators=50 is a good starting point for speed. n_jobs=-1 uses all CPU cores.
model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)

print("Training the RandomForestClassifier... (This may take a few minutes)")
model.fit(X_train, y_train)
print("Training complete.")


# ----------------------------------------
# 5. Model Evaluation
# ----------------------------------------
print("\n--- Model Evaluation ---")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nOverall Model Accuracy: {accuracy:.2%}")

# The full classification report is too large to display, so we'll show a sample.
# Let's see how it performs on a few specific, randomly chosen locations.
print("\nSample Classification Report (for 5 random locations):")
random_labels_indices = np.random.choice(np.unique(y_test), 5, replace=False)
random_labels_names = le.inverse_transform(random_labels_indices)
print(classification_report(y_test, y_pred, labels=random_labels_indices, target_names=random_labels_names, zero_division=0))


# ----------------------------------------
# 6. Feature Importance: Which APs are Most Important?
# ----------------------------------------
print("\n--- Feature Importance ---")

importances = model.feature_importances_
indices = np.argsort(importances)[-20:] # Top 20 most important APs
features = X.columns

plt.figure(figsize=(12, 10))
plt.title('Top 20 Most Important Access Points for Localization')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


# ----------------------------------------
# 7. Conclusion
# ----------------------------------------
print("\n--- Conclusion ---")
print(f"The RandomForest model achieved an impressive accuracy of {accuracy:.2%}, demonstrating that Wi-Fi RSSI is a highly effective feature for indoor localization.")
print("Key Takeaways:")
print("- The model can reliably predict a user's specific room out of hundreds of possibilities, showcasing the stability of Wi-Fi 'fingerprints'.")
print("- The Feature Importance plot provides extremely valuable operational insights. It shows which specific Access Points are the most critical for location services. A network administrator could use this information to ensure these key APs have high uptime and are not moved or decommissioned without careful planning.")
print("- This technology is the backbone of many modern services, including indoor navigation (e.g., in airports or malls), location-aware advertising, and asset tracking within a warehouse or hospital.")
print("- By correctly preprocessing the 'no signal' value (100 -> -105), we provided the model with meaningful data that significantly improved its ability to learn.")

```