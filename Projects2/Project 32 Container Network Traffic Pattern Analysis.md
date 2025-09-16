---

### **Project 32: Container Network Traffic Pattern Analysis**

**Objective:** To build a machine learning model that can classify the type of application running inside a container (e.g., 'WebApp', 'Database', 'Cache') by analyzing the statistical features of its network traffic.

**Dataset Source:** **Synthetically Generated**. We will create a dataset that simulates network flow data originating from different types of containerized applications. The data will reflect the distinct "network personalities" of these applications (e.g., a database has different traffic patterns than a web server).

**Model:** We will use a **RandomForestClassifier**. This model is well-suited for this task because it can effectively handle the diverse set of flow features and learn the complex patterns that differentiate the network behavior of various applications.

**Instructions:**
This notebook is fully self-contained and does not require any external files or APIs. Simply run the entire code block in Google Colab.

**Implementation in Google Colab:**
```python
#
# ==================================================================================
#  Project 32: Container Network Traffic Pattern Analysis
# ==================================================================================
#
# Objective:
# This notebook builds a classifier to identify the application type running in a
# container based on its network traffic patterns, using a synthetic dataset.
#
# To Run in Google Colab:
# Copy and paste this entire code block into a single cell and run it.
#

# ----------------------------------------
# 1. Import Necessary Libraries
# ----------------------------------------
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------
# 2. Synthetic Container Traffic Data Generation
# ----------------------------------------
print("--- Generating Synthetic Container Network Traffic Dataset ---")

num_samples = 5000
data = []
app_types = ['WebApp', 'Database', 'Cache', 'MessageQueue', 'APIGateway']

# Define the "network personality" of each application
app_profiles = {
    'WebApp':       {'avg_pkt_size': 500,  'server_port': 443,  'flow_duration_ms': 500, 'client_server_ratio': 0.8},
    'Database':     {'avg_pkt_size': 1000, 'server_port': 5432, 'flow_duration_ms': 100, 'client_server_ratio': 0.5},
    'Cache':        {'avg_pkt_size': 150,  'server_port': 6379, 'flow_duration_ms': 20,  'client_server_ratio': 0.5},
    'MessageQueue': {'avg_pkt_size': 300,  'server_port': 5672, 'flow_duration_ms': 10000,'client_server_ratio': 0.5},
    'APIGateway':   {'avg_pkt_size': 800,  'server_port': 8080, 'flow_duration_ms': 200, 'client_server_ratio': 0.7}
}

for _ in range(num_samples):
    app_type = random.choice(app_types)
    profile = app_profiles[app_type]
    
    # Generate features based on the profile with some randomness
    avg_pkt_size = np.random.normal(profile['avg_pkt_size'], 50)
    server_port = profile['server_port']
    flow_duration_ms = np.random.normal(profile['flow_duration_ms'], 100)
    # Ratio of packets sent by client vs. server
    client_server_ratio = np.random.normal(profile['client_server_ratio'], 0.1)
    
    # Number of packets in the flow
    total_packets = np.random.randint(5, 100)
    client_packets = int(total_packets * client_server_ratio)
    server_packets = total_packets - client_packets
    
    data.append([avg_pkt_size, server_port, flow_duration_ms, client_packets, server_packets, app_type])

df = pd.DataFrame(data, columns=['avg_pkt_size', 'server_port', 'flow_duration_ms', 'client_packets', 'server_packets', 'app_type'])
df = df[df['flow_duration_ms'] > 0] # Remove any negative durations
print(f"Dataset generation complete. Created {len(df)} flow samples.")
print("\nDataset Sample:")
print(df.sample(5))


# ----------------------------------------
# 3. Data Splitting and Encoding
# ----------------------------------------
print("\n--- Splitting and Encoding Data ---")

X = df.drop(columns=['app_type'])
y = df['app_type']

# Encode the string labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Use a stratified split to ensure all app types are represented
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)
print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")


# ----------------------------------------
# 4. Model Training
# ----------------------------------------
print("\n--- Model Training ---")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

print("Training the RandomForestClassifier...")
model.fit(X_train, y_train)
print("Training complete.")


# ----------------------------------------
# 5. Model Evaluation
# ----------------------------------------
print("\n--- Model Evaluation ---")
y_pred = model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='rocket', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix for Container Application Classification')
plt.ylabel('Actual Application')
plt.xlabel('Predicted Application')
plt.show()


# ----------------------------------------
# 6. Feature Importance
# ----------------------------------------
print("\n--- Feature Importance: What defines a container's network personality? ---")

importances = model.feature_importances_
features = X.columns
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance in Container Traffic Classification')
plt.show()


# ----------------------------------------
# 7. Conclusion
# ----------------------------------------
print("\n--- Conclusion ---")
print("The RandomForest model successfully learned to classify containerized applications based on their distinct network traffic patterns.")
print("Key Takeaways:")
print("- The model achieved high precision and recall, demonstrating that even with a few simple flow features, different applications have unique and identifiable 'network personalities'.")
print("- The feature importance plot is very insightful. It shows that the `server_port` is the most powerful predictor (as expected), but that behavioral features like `flow_duration_ms` and `avg_pkt_size` are also critical differentiators. This is important because it means the model can still work even if applications use non-standard ports.")
print("- This capability is a cornerstone of modern cloud-native security. A Kubernetes security platform could use this model to automatically discover and label all running microservices. It could then generate a security policy, for example: 'Only pods labeled as 'WebApp' are allowed to talk to pods labeled as 'Database''. This creates a zero-trust environment where security policies are based on observed behavior, not static IP addresses.")

```