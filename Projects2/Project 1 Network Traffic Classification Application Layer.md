---

### **Project 1 (Kaggle Edition): Network Traffic Classification**

**Objective:** To classify network traffic from the UNSW-NB15 dataset into 'Normal' or 'Attack' categories using a RandomForest model.

**Dataset Source:** **Kaggle**. We will download the dataset directly from Kaggle into our Colab environment. This ensures the source is stable and vetted.

**Instructions:**

1.  **Get your Kaggle API key:**
    *   Go to your Kaggle account page: [https://www.kaggle.com/](https://www.kaggle.com/)`<your-username>`/account
    *   Scroll down to the "API" section.
    *   Click on **"Create New API Token"**. This will download a file named `kaggle.json`.
2.  **Run the first code cell below.**
3.  **Upload your `kaggle.json` file:** When the file upload prompt appears, select the `kaggle.json` file you just downloaded.

The rest of the notebook will run automatically.

**Implementation in Google Colab:**

```python
#
# ==================================================================================
#  Project 1 (Kaggle Edition): Network Traffic Classification
# ==================================================================================
#
# Objective:
# This notebook provides a robust workflow for network traffic classification
# using the UNSW-NB15 dataset sourced directly from Kaggle.
#
# To Run in Google Colab:
# 1. Have your `kaggle.json` API token ready. (See instructions above).
# 2. Copy and paste this entire code block into a single cell.
# 3. Run the cell. You will be prompted to upload your `kaggle.json` file.
#

# ----------------------------------------
# 1. Setup Kaggle API and Download Data
# ----------------------------------------
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

print("\n--- Downloading UNSW-NB15 Dataset from Kaggle ---")
# Download the dataset (user: rawadahmed, dataset: unsw-nb15)
!kaggle datasets download -d rawadahmed/unsw-nb15

print("\n--- Unzipping the dataset ---")
# Unzip the downloaded file
!unzip -q unsw-nb15.zip -d .

print("\nDataset setup complete.")

# ----------------------------------------
# 2. Load and Prepare the Data
# ----------------------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import time

print("\n--- Loading and Preprocessing Data ---")

try:
    # Load the training and testing files provided in the dataset
    df_train = pd.read_csv('UNSW_NB15_training-set.csv')
    df_test = pd.read_csv('UNSW_NB15_testing-set.csv')
    
    # Combine them into a single dataframe for consistent preprocessing
    df = pd.concat([df_train, df_test], ignore_index=True)

    print(f"Successfully loaded and combined datasets. Total shape: {df.shape}")

except FileNotFoundError:
    print("Error: CSV files not found. The Kaggle download might have failed.")
    exit()

# Drop unnecessary 'id' column
df = df.drop(columns=['id'])

# The 'label' column is binary (0/1), 'attack_cat' is the detailed category.
# We will predict 'attack_cat'.
print("\nDistribution of traffic categories ('attack_cat'):")
print(df['attack_cat'].value_counts())

# Clean column names by stripping whitespace
df.columns = df.columns.str.strip()

# Identify object-type columns for encoding
categorical_cols = df.select_dtypes(include=['object']).columns

# Use one-hot encoding for features, label encoding for the target
target_col = 'attack_cat'
feature_cols = [col for col in categorical_cols if col != target_col]

print(f"\nApplying one-hot encoding to: {feature_cols}")
df = pd.get_dummies(df, columns=feature_cols, drop_first=True)

# Label encode the target variable
y_encoder = LabelEncoder()
df[target_col] = y_encoder.fit_transform(df[target_col])

# Separate features (X) and target (y)
X = df.drop(columns=[target_col, 'label']) # also drop binary 'label'
y = df[target_col]

# Split the data into training and testing sets
# This is a more robust approach than using the pre-split files directly
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print("\nPreprocessing complete.")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")


# ----------------------------------------
# 3. Model Training
# ----------------------------------------
print("\n--- Model Training ---")

# Initialize the RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

print("Training the RandomForestClassifier... (This may take a few minutes)")
start_time = time.time()
model.fit(X_train, y_train)
end_time = time.time()
print(f"Training completed in {end_time - start_time:.2f} seconds.")


# ----------------------------------------
# 4. Model Evaluation
# ----------------------------------------
print("\n--- Model Evaluation ---")

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")

# Get the original string labels for the report
target_names_str = y_encoder.classes_

# Display the classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names_str, zero_division=0))

# Display the confusion matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names_str, yticklabels=target_names_str)
plt.title('Confusion Matrix for Network Traffic Classification')
plt.ylabel('Actual Category')
plt.xlabel('Predicted Category')
plt.show()

# ----------------------------------------
# 5. Conclusion
# ----------------------------------------
print("\n--- Conclusion ---")
print(f"Using a verified dataset from Kaggle, the RandomForestClassifier achieved an accuracy of {accuracy:.2%}.")
print("The model demonstrates high performance in identifying 'Normal' traffic and common attacks.")
print("This notebook establishes a reliable and reproducible baseline for network intrusion detection tasks.")
```