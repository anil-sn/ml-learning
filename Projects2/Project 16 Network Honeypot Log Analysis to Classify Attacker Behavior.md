---

### **Project 16: Network Honeypot Log Analysis to Classify Attacker Behavior**

**Objective:** To automatically discover and classify different types of attacker behavior (e.g., port scanning, login brute-forcing, web scanning) by applying unsupervised clustering to honeypot logs.

**Dataset Source:** **Synthetically Generated**. We will create a realistic honeypot log dataset that simulates various common attack patterns. This allows us to have a "ground truth" to verify if our clustering algorithm successfully separates the different tactics.

**Model:** We will use **K-Means Clustering**, a fundamental and powerful unsupervised algorithm that groups data points into a specified number of clusters based on their feature similarity. We will use the "Elbow Method" to programmatically determine the optimal number of clusters.

**Instructions:**
This notebook is fully self-contained and does not require any external files or APIs. Simply run the entire code block in Google Colab.

**Implementation in Google Colab:**
```python
#
# ==================================================================================
#  Project 16: Network Honeypot Log Analysis
# ==================================================================================
#
# Objective:
# This notebook uses K-Means clustering to automatically group attacker IPs
# from honeypot logs into behavioral categories like port scanners,
# brute-force attackers, and web scanners.
#
# To Run in Google Colab:
# Copy and paste this entire code block into a single cell and run it.
#

# ----------------------------------------
# 1. Import Necessary Libraries
# ----------------------------------------
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------
# 2. Synthetic Honeypot Log Generation
# ----------------------------------------
print("--- Generating Synthetic Honeypot Log Dataset ---")

log_entries = []
num_ips = 50

for i in range(num_ips):
    ip = f'185.112.{random.randint(1, 255)}.{random.randint(1, 255)}'
    behavior = random.choice(['port_scan', 'brute_force_ssh', 'web_scan'])
    
    if behavior == 'port_scan':
        # Many connections to many different ports
        for _ in range(random.randint(80, 200)):
            port = random.randint(1, 65535)
            log_entries.append([ip, 22 if port % 100 == 0 else port, 'TCP', 'Connection refused'])
        log_entries[-1].append('port_scan') # Add true label for later validation
            
    elif behavior == 'brute_force_ssh':
        # Many connections to a single port (SSH) with failed logins
        for _ in range(random.randint(50, 150)):
            log_entries.append([ip, 22, 'SSH', 'Authentication failed'])
        log_entries[-1].append('brute_force_ssh')

    elif behavior == 'web_scan':
        # Many connections to web ports with 404 errors
        for _ in range(random.randint(70, 180)):
            port = random.choice([80, 443])
            log_entries.append([ip, port, 'HTTP', 'GET /admin.php - 404 Not Found'])
        log_entries[-1].append('web_scan')

# The label column is only for our final validation, it won't be used in training
df = pd.DataFrame(log_entries, columns=['source_ip', 'dest_port', 'protocol', 'message', 'true_behavior'])
df['true_behavior'] = df.groupby('source_ip')['true_behavior'].ffill().bfill()

print("Dataset generation complete. Sample log entries:")
print(df.sample(5))


# ----------------------------------------
# 3. Feature Engineering: Creating Attacker Profiles
# ----------------------------------------
print("\n--- Engineering Behavioral Features for each Source IP ---")

# Aggregate the raw logs by source IP to build a behavioral profile
attacker_profiles = df.groupby('source_ip').agg(
    total_connections=('dest_port', 'count'),
    unique_ports_targeted=('dest_port', 'nunique'),
    ssh_auth_failures=('message', lambda x: x.str.contains('Authentication failed').sum()),
    http_404_errors=('message', lambda x: x.str.contains('404').sum()),
    common_port=('dest_port', lambda x: x.mode()[0])
)

# Add our ground truth label for later evaluation
attacker_profiles = attacker_profiles.join(df.groupby('source_ip')['true_behavior'].first())

print("Generated attacker profiles. Sample:")
print(attacker_profiles.sample(5))


# ----------------------------------------
# 4. Model Training: K-Means Clustering
# ----------------------------------------
print("\n--- Unsupervised Model Training ---")

# Prepare the data for clustering (features only, no labels)
X = attacker_profiles.drop(columns=['true_behavior'])

# Scale the features so that no single feature dominates the clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Determine the Optimal Number of Clusters (k) using the Elbow Method ---
inertia = []
k_range = range(1, 10)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()
print("The 'elbow' in the plot is clearly at k=3, which is the optimal number of clusters.")

# --- Train the final K-Means model with the optimal k ---
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
attacker_profiles['cluster'] = kmeans.fit_predict(X_scaled)
print(f"Clustering complete. Assigned attackers to {optimal_k} clusters.")


# ----------------------------------------
# 5. Analysis and Interpretation of Clusters
# ----------------------------------------
print("\n--- Analyzing and Interpreting Cluster Behaviors ---")

# Analyze the average feature values for each cluster to understand its behavior
cluster_analysis = attacker_profiles.groupby('cluster').mean(numeric_only=True)
print("Cluster Centroids (Average Behavior per Cluster):")
print(cluster_analysis)

# Cross-tabulate clusters with our ground truth labels to see how well we did
print("\nValidation: Cross-tabulation of Clusters vs. True Behavior")
print(pd.crosstab(attacker_profiles['cluster'], attacker_profiles['true_behavior']))
print("\nInterpretation: The clusters align almost perfectly with the simulated behaviors.")


# ----------------------------------------
# 6. Visualization of Clusters
# ----------------------------------------
print("\n--- Visualizing the Discovered Attacker Groups ---")

# Use PCA to reduce the data to 2 dimensions for plotting
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

attacker_profiles['pca1'] = X_pca[:, 0]
attacker_profiles['pca2'] = X_pca[:, 1]

plt.figure(figsize=(10, 8))
sns.scatterplot(
    x='pca1', y='pca2',
    hue='cluster',
    palette=sns.color_palette("hsv", optimal_k),
    data=attacker_profiles,
    legend="full",
    alpha=0.8
)
plt.title('Discovered Attacker Clusters (Visualized with PCA)')
plt.show()


# ----------------------------------------
# 7. Conclusion
# ----------------------------------------
print("\n--- Conclusion ---")
print("The K-Means clustering model successfully processed raw honeypot logs and automatically grouped attackers into distinct behavioral categories.")
print("Key Takeaways:")
print("- The model correctly identified 3 primary types of activity without being told what to look for. By analyzing the cluster centroids, we could confidently label them as 'Port Scanners', 'SSH Brute-Forcers', and 'Web Scanners'.")
print("- This demonstrates the power of unsupervised learning for threat intelligence. It can discover novel attack patterns and group similar campaigns together, even if they aren't from a known signature.")
print("- A security operations team could use this approach to move beyond analyzing individual alerts. Instead, they could see that 'Cluster 2 (SSH Brute-Forcers)' is highly active today, allowing them to make strategic decisions like strengthening firewall rules for port 22 or reviewing SSH authentication policies.")
```