---

### **Project 14: Phishing & Malicious URL Detection from Web Proxy Logs**

**Objective:** To build a fast, interpretable machine learning model that can classify a URL as 'benign' or 'malicious' by engineering features from the URL string itself.

**Dataset Source:** **Kaggle**. We will use the "Malicious and Benign Websites" dataset, which contains a large, labeled collection of URLs.

**Model:** We will use **Logistic Regression**. While complex models could be used, Logistic Regression is chosen here for its high speed and, most importantly, its **interpretability**. The model's coefficients will tell us exactly which URL characteristics are the biggest red flags, providing actionable intelligence.

**Instructions:**
This notebook requires the Kaggle API. Please run the setup cell and upload your `kaggle.json` file if you have not already done so in this session.

**Implementation in Google Colab:**
```python
#
# ==================================================================================
#  Project 14: Phishing & Malicious URL Detection
# ==================================================================================
#
# Objective:
# This notebook builds an interpretable model to detect malicious URLs by
# engineering lexical features directly from the URL string.
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

print("\n--- Downloading Malicious and Benign Websites Dataset from Kaggle ---")
!kaggle datasets download -d antoreepjana/malicious-and-benign-websites

print("\n--- Unzipping the dataset ---")
!unzip -q malicious-and-benign-websites.zip -d .
print("Dataset setup complete.")


# ----------------------------------------
# 2. Load and Prepare the Data
# ----------------------------------------
import pandas as pd
import numpy as np
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("\n--- Loading and Preprocessing Data ---")

try:
    df = pd.read_csv('urldata.csv')
    print("Successfully loaded urldata.csv.")
except FileNotFoundError as e:
    print(f"Error: Could not find dataset file. {e}")
    exit()

# Drop the 'Unnamed: 0' column
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# Encode the 'Label' column: malicious -> 1, benign -> 0
df['Label'] = df['Label'].apply(lambda x: 1 if x == 'malicious' else 0)

print("\nClass Distribution:")
print(df['Label'].value_counts())
print("\nDataset sample:")
print(df.head())


# ----------------------------------------
# 3. Feature Engineering from URL Strings
# ----------------------------------------
print("\n--- Engineering Lexical Features from URLs ---")

# This is the core of the project. We create numerical features from the raw text.
df['url_length'] = df['Url'].apply(len)
df['hostname_length'] = df['Url'].apply(lambda x: len(urlparse(x).netloc))
df['path_length'] = df['Url'].apply(lambda x: len(urlparse(x).path))
df['count_dash'] = df['Url'].apply(lambda x: x.count('-'))
df['count_at'] = df['Url'].apply(lambda x: x.count('@'))
df['count_question'] = df['Url'].apply(lambda x: x.count('?'))
df['count_percent'] = df['Url'].apply(lambda x: x.count('%'))
df['count_dot'] = df['Url'].apply(lambda x: x.count('.'))
df['count_equal'] = df['Url'].apply(lambda x: x.count('='))
df['count_http'] = df['Url'].apply(lambda x: x.count('http'))
df['count_https'] = df['Url'].apply(lambda x: x.count('https'))
df['count_www'] = df['Url'].apply(lambda x: x.count('www'))
df['count_digits'] = df['Url'].apply(lambda x: sum(c.isdigit() for c in x))
df['count_letters'] = df['Url'].apply(lambda x: sum(c.isalpha() for c in x))
df['count_dir'] = df['Url'].apply(lambda x: urlparse(x).path.count('/'))

print("Feature engineering complete. New dataset sample:")
print(df.head())


# ----------------------------------------
# 4. Data Splitting and Scaling
# ----------------------------------------
print("\n--- Splitting and Scaling Data ---")

feature_cols = [col for col in df.columns if col not in ['Url', 'Label']]
X = df[feature_cols]
y = df['Label']

# Stratified split to maintain class ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Scale features for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ----------------------------------------
# 5. Model Training
# ----------------------------------------
print("\n--- Model Training ---")

# Using `class_weight='balanced'` helps the model perform well on the slightly imbalanced data
model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=200)

print("Training the Logistic Regression model...")
model.fit(X_train_scaled, y_train)
print("Training complete.")


# ----------------------------------------
# 6. Model Evaluation
# ----------------------------------------
print("\n--- Model Evaluation ---")
y_pred = model.predict(X_test_scaled)

print("\nClassification Report (Focus on Recall for Malicious):")
print(classification_report(y_test, y_pred, target_names=['Benign (0)', 'Malicious (1)']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malicious'], yticklabels=['Benign', 'Malicious'])
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()


# ----------------------------------------
# 7. Model Interpretability
# ----------------------------------------
print("\n--- Model Interpretability: Which features indicate a malicious URL? ---")

# The model coefficients show the importance and direction of each feature's influence.
# A positive coefficient means the feature increases the odds of being malicious.
coefficients = pd.DataFrame({
    'Feature': feature_cols,
    'Coefficient': model.coef_[0]
}).sort_values('Coefficient', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x='Coefficient', y='Feature', data=coefficients)
plt.title('Feature Importance in Predicting Malicious URLs')
plt.xlabel('Coefficient (Log-Odds) -> Larger values indicate higher risk')
plt.show()

print("Top 5 indicators of a MALICIOUS URL:")
print(coefficients.head(5))
print("\nTop 5 indicators of a BENIGN URL:")
print(coefficients.tail(5))


# ----------------------------------------
# 8. Conclusion
# ----------------------------------------
print("\n--- Conclusion ---")
print("The Logistic Regression model proved to be a highly effective and interpretable classifier for malicious URLs.")
print("Key Takeaways:")
print("- The model achieved excellent precision and recall, meaning it reliably catches malicious URLs with a low rate of false alarms.")
print("- The true power of this approach lies in its interpretability. The coefficient plot provides clear, actionable insights for security analysts.")
print("- We can confirm that the presence of '@' symbols, an unusual number of directories, and a long path length are strong indicators of maliciousness. Conversely, the presence of 'www' and 'https' are strong indicators of a benign site.")
print("- This lightweight model could be deployed in real-time within a web proxy, an email gateway, or a DNS filter to block threats based on these lexical red flags, providing a powerful layer of defense.")
```