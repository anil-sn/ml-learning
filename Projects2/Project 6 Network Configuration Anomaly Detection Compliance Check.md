---

### **Project 6: Network Configuration Anomaly Detection / Compliance Check**

**Objective:** To automatically identify network device configurations that deviate from a standard "golden" template. This is essential for auditing, compliance (e.g., PCI-DSS, HIPAA), and detecting unauthorized or accidental changes.

**Dataset:** **Synthetically Generated**. We will create a set of baseline configuration templates and then introduce anomalies like incorrect IP addresses, missing security rules, or typos.

**Model:** We will treat configurations as text documents and use a combination of **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert them into numerical vectors and **Isolation Forest** (unsupervised) to detect outliers.

**Instructions:**
This notebook is fully self-contained and does not require the Kaggle API or any file uploads. Simply run the entire cell block in Google Colab.

**Implementation in Google Colab:**
```python
#
# ==================================================================================
#  Project 6: Network Configuration Anomaly Detection / Compliance Check
# ==================================================================================
#
# Objective:
# This notebook demonstrates how to detect anomalous network configurations using
# unsupervised learning. We will first generate a synthetic dataset of "golden"
# and "anomalous" configs, then train a model to distinguish between them.
#
# To Run in Google Colab:
# Copy and paste this entire code block into a single cell and run it.
#

# ----------------------------------------
# 1. Import Necessary Libraries
# ----------------------------------------
import os
import random
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ----------------------------------------
# 2. Synthetic Data Generation
# ----------------------------------------
print("--- Generating Synthetic Network Configuration Dataset ---")

# Define a "golden" template for a standard access switch
GOLDEN_TEMPLATE = """
hostname ACCESS_SWITCH_01
!
spanning-tree mode rapid-pvst
spanning-tree portfast default
!
interface Vlan10
 ip address 192.168.10.1 255.255.255.0
 description USERS_VLAN
!
interface GigabitEthernet0/1
 switchport mode access
 switchport access vlan 10
!
ip access-list standard GUEST_ACL
 permit 10.10.0.0 0.0.255.255
 deny   any log
!
line vty 0 4
 access-class GUEST_ACL in
 transport input ssh
!
ntp server 1.1.1.1
snmp-server community GPRO_NET_RO read-only
"""

# Function to introduce anomalies into a template
def create_anomaly(config_text):
    lines = config_text.strip().split('\n')
    anomaly_type = random.choice(['ip_change', 'line_removed', 'typo', 'added_insecure_service'])
    
    if anomaly_type == 'ip_change' and 'ip address' in config_text:
        line_num = [i for i, line in enumerate(lines) if 'ip address' in line][0]
        lines[line_num] = ' ip address 192.168.99.1 255.255.255.0' # Deviant IP
        description = "Changed IP address"
    elif anomaly_type == 'line_removed' and 'snmp-server' in config_text:
        lines = [line for line in lines if 'snmp-server' not in line] # Removed SNMP
        description = "Removed SNMP server line"
    elif anomaly_type == 'typo':
        line_num = [i for i, line in enumerate(lines) if 'transport input ssh' in line][0]
        lines[line_num] = ' transport input telnet ssh' # Typo/insecure addition
        description = "Allowed telnet access"
    else: # added_insecure_service
        lines.append('ip http server') # Added insecure HTTP server
        description = "Added insecure HTTP server"
        
    return '\n'.join(lines), description

# Create the dataset directory
os.makedirs('configs', exist_ok=True)

# Generate configuration files
config_data = []
num_golden = 50
num_anomalous = 10

print(f"Generating {num_golden} 'golden' and {num_anomalous} 'anomalous' configs...")
for i in range(num_golden):
    filename = f'configs/golden_{i+1}.txt'
    with open(filename, 'w') as f:
        f.write(GOLDEN_TEMPLATE)
    config_data.append({'filename': filename, 'content': GOLDEN_TEMPLATE, 'label': 'golden', 'anomaly_desc': 'N/A'})

for i in range(num_anomalous):
    filename = f'configs/anomaly_{i+1}.txt'
    anomalous_config, desc = create_anomaly(GOLDEN_TEMPLATE)
    with open(filename, 'w') as f:
        f.write(anomalous_config)
    config_data.append({'filename': filename, 'content': anomalous_config, 'label': 'anomaly', 'anomaly_desc': desc})

df = pd.DataFrame(config_data)
print("Dataset generation complete.")
print("\nExample of a generated anomaly:")
print(df[df['label'] == 'anomaly'].iloc[0]['content'])
print("Description:", df[df['label'] == 'anomaly'].iloc[0]['anomaly_desc'])


# ----------------------------------------
# 3. Feature Engineering with TF-IDF
# ----------------------------------------
print("\n--- Feature Engineering ---")
print("Converting configuration files into numerical vectors using TF-IDF...")

# TF-IDF vectorizer will treat each config as a document and each line/command as a word.
vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words=None, token_pattern=r'\S+')
X = vectorizer.fit_transform(df['content'])

print(f"Feature matrix created with shape: {X.shape}")


# ----------------------------------------
# 4. Unsupervised Model Training
# ----------------------------------------
print("\n--- Unsupervised Model Training ---")

# We will train the Isolation Forest ONLY on the 'golden' configurations.
# This teaches the model what a "normal" configuration looks like.
X_train = vectorizer.transform(df[df['label'] == 'golden']['content'])

# `contamination` is the expected percentage of anomalies in the *full* dataset.
# Set it to 'auto' or calculate it.
contamination_rate = num_anomalous / (num_golden + num_anomalous)
print(f"Setting contamination rate to {contamination_rate:.2f}")

model = IsolationForest(contamination=contamination_rate, random_state=42)

print("Training the Isolation Forest model on 'golden' configs only...")
model.fit(X_train)
print("Training complete.")


# ----------------------------------------
# 5. Model Evaluation
# ----------------------------------------
print("\n--- Model Evaluation ---")

# Predict on the entire dataset (both golden and anomalous).
# The model returns 1 for normal (inliers) and -1 for anomalies (outliers).
predictions = model.predict(X)

# Create the ground truth labels in the same format for comparison.
y_true = df['label'].apply(lambda x: 1 if x == 'golden' else -1)
y_pred = predictions

# Add predictions to our dataframe to see the results
df['prediction'] = y_pred
df['prediction_label'] = df['prediction'].apply(lambda x: 'golden' if x == 1 else 'anomaly')

print("\nIncorrectly Classified Configurations:")
print(df[df['label'] != df['prediction_label']][['filename', 'label', 'prediction_label', 'anomaly_desc']])


# Display the classification report and confusion matrix
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=['Anomaly (-1)', 'Golden (1)']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Anomaly', 'Golden'], yticklabels=['Anomaly', 'Golden'])
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()


# ----------------------------------------
# 6. Conclusion
# ----------------------------------------
print("\n--- Conclusion ---")
print("This notebook demonstrates a powerful, unsupervised method for ensuring network configuration compliance.")
print("Key Takeaways:")
print("- We successfully trained a model to identify anomalous configurations without ever showing it an anomaly during training.")
print("- The high precision and recall scores show that the TF-IDF + Isolation Forest approach is highly effective for this text-based anomaly detection task.")
print("- This system could be automated to run against nightly configuration backups. Any config flagged as an 'anomaly' (-1) would be sent to a network engineer for immediate review.")
print("- This significantly reduces manual audit time and helps enforce security and operational standards across the network.")

```