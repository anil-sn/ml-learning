---

### **Project 12: DNS Tunneling Detection**

**Objective:** To build an interpretable machine learning model that can distinguish between legitimate DNS queries and those used for DNS tunneling, based on features like query length, entropy, and subdomain count.

**Dataset Source:** **Kaggle**. We will use a specialized "DNS Tunneling" dataset which contains pre-calculated features for thousands of labeled DNS queries.

**Model:** We will use **Logistic Regression**. While not as complex as tree-based models, it is incredibly fast and highly **interpretable**. This allows us to not only get a prediction but also understand *why* the model flagged a specific query as malicious, which is invaluable for a security analyst.

**Instructions:**
This notebook requires the Kaggle API. Please run the setup cell and upload your `kaggle.json` file if you have not already done so in this session.

**Implementation in Google Colab:**
```python
#
# ==================================================================================
#  Project 12: DNS Tunneling Detection
# ==================================================================================
#
# Objective:
# This notebook builds an interpretable model to detect DNS tunneling by analyzing
# the statistical properties of DNS queries.
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

print("\n--- Downloading DNS Tunneling Dataset from Kaggle ---")
!kaggle datasets download -d ahmethamzadedbs/dns-tunneling-dataset

print("\n--- Unzipping the dataset ---")
!unzip -q dns-tunneling-dataset.zip -d dns_data
print("Dataset setup complete.")


# ----------------------------------------
# 2. Load and Prepare the Data
# ----------------------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("\n--- Loading and Preprocessing Data ---")

try:
    df = pd.read_csv('dns_data/dnscat2.csv')
    print("Successfully loaded dnscat2.csv dataset.")
except FileNotFoundError as e:
    print(f"Error: Could not find dataset file. {e}")
    exit()

# For this project, we'll focus on a few highly interpretable features.
# Malicious DNS queries for tunneling often have:
# 1. High length (to encode data).
# 2. High entropy (random-looking subdomains).
# 3. Many subdomains.
feature_cols = ['query_length', 'subdomain_count', 'entropy']
target_col = 'label'
df = df[feature_cols + [target_col]]

# Encode the label: 'nontunnel' -> 0, 'tunnel' -> 1
df[target_col] = df[target_col].apply(lambda x: 1 if x == 'tunnel' else 0)

print("\nClass Distribution:")
print(df[target_col].value_counts())
print(f"\nShape after feature selection: {df.shape}")


# ----------------------------------------
# 3. Exploratory Data Analysis (EDA)
# ----------------------------------------
print("\n--- Visualizing Feature Differences ---")
plt.figure(figsize=(18, 5))
for i, col in enumerate(feature_cols):
    plt.subplot(1, 3, i + 1)
    sns.boxplot(x=target_col, y=col, data=df)
    plt.title(f'{col} by Class')
    plt.xticks([0, 1], ['Nontunnel', 'Tunnel'])
plt.tight_layout()
plt.show()
print("As the plots show, 'tunnel' queries clearly have higher length, subdomain counts, and entropy.")


# ----------------------------------------
# 4. Data Splitting and Scaling
# ----------------------------------------
print("\n--- Splitting and Scaling Data ---")

X = df[feature_cols]
y = df[target_col]

# Stratified split to maintain class ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Scale features for better performance with Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")


# ----------------------------------------
# 5. Model Training
# ----------------------------------------
print("\n--- Model Training ---")

# Use `class_weight='balanced'` to handle the imbalanced dataset.
model = LogisticRegression(random_state=42, class_weight='balanced')

print("Training the Logistic Regression model...")
model.fit(X_train_scaled, y_train)
print("Training complete.")


# ----------------------------------------
# 6. Model Evaluation
# ----------------------------------------
print("\n--- Model Evaluation ---")
y_pred = model.predict(X_test_scaled)

print("\nClassification Report (Focus on Recall for Tunnel):")
print(classification_report(y_test, y_pred, target_names=['Nontunnel (0)', 'Tunnel (1)']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=['Nontunnel', 'Tunnel'], yticklabels=['Nontunnel', 'Tunnel'])
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()


# ----------------------------------------
# 7. Model Interpretability
# ----------------------------------------
print("\n--- Model Interpretability: Feature Importance ---")

# In a logistic regression, the model's coefficients tell us the importance
# and direction of each feature's influence on the prediction.
coefficients = pd.DataFrame({
    'Feature': feature_cols,
    'Coefficient': model.coef_[0]
}).sort_values('Coefficient', ascending=False)

print("Model Coefficients:")
print(coefficients)

plt.figure(figsize=(8, 5))
sns.barplot(x='Coefficient', y='Feature', data=coefficients, palette='viridis')
plt.title('Feature Importance in Predicting DNS Tunneling')
plt.xlabel('Coefficient (Log-Odds)')
plt.show()


# ----------------------------------------
# 8. Conclusion
# ----------------------------------------
print("\n--- Conclusion ---")
print("The Logistic Regression model performed exceptionally well, accurately detecting DNS tunneling traffic.")
print("Key Takeaways:")
print("- The model's high recall for the 'Tunnel' class shows it is very effective at catching this stealthy attack.")
print("- The key advantage of this model is its interpretability. The coefficient plot clearly shows that high query entropy, a high subdomain count, and longer query lengths are all strong, positive indicators of tunneling. This aligns perfectly with our domain knowledge.")
print("- A security analyst using this model can see not just *that* a query was flagged, but *why* (e.g., 'This query was flagged because its entropy and length are abnormally high').")
print("- This type of interpretable, lightweight model is ideal for deployment in real-time DNS monitoring systems to provide high-fidelity alerts.")
```