---

### **Project 17: Wi-Fi Anomaly Detection (Deauthentication Flood)**

**Objective:** To build an unsupervised anomaly detection system that can identify a deauthentication flood attack in real-time by analyzing the rate and type of Wi-Fi management frames.

**Dataset Source:** **Synthetically Generated**. We will create a time-series dataset that simulates the capture of Wi-Fi management frames, first establishing a baseline of normal activity and then introducing a deauthentication attack. This is a common requirement as real-world, labeled Wi-Fi attack captures are not always available.

**Model:** We will use the **Isolation Forest** algorithm. It is an excellent choice for this task because it can learn the characteristics of "normal" network behavior and then flag periods of time that deviate significantly from that baseline, without needing prior examples of the attack.

**Instructions:**
This notebook is fully self-contained and does not require any external files or APIs. Simply run the entire code block in Google Colab.

**Implementation in Google Colab:**
```python
#
# ==================================================================================
#  Project 17: Wi-Fi Anomaly Detection (Deauthentication Flood)
# ==================================================================================
#
# Objective:
# This notebook builds an unsupervised model to detect a Wi-Fi deauthentication
# flood attack by analyzing the statistical properties of management frames over time.
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
import time
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------
# 2. Synthetic Wi-Fi Frame Generation
# ----------------------------------------
print("--- Generating Synthetic Wi-Fi Frame Dataset ---")

# Simulation parameters
total_duration_seconds = 120
attack_start_time = 80
attack_duration = 20
normal_frames_per_second = 50
attack_frames_per_second = 500

# Lists of possible frame subtypes
normal_subtypes = ['Beacon', 'Probe Request', 'Probe Response', 'Association Request', 'Deauthentication']
# During normal operation, deauth frames are rare (e.g., a user manually disconnects)
normal_subtype_weights = [0.5, 0.2, 0.2, 0.09, 0.01]

# Generate the data second by second
timestamps = []
frame_subtypes = []

for second in range(total_duration_seconds):
    current_time = time.time()
    
    if attack_start_time <= second < attack_start_time + attack_duration:
        # --- ATTACK PERIOD ---
        num_frames = attack_frames_per_second
        # During an attack, the vast majority of frames are deauthentication frames
        subtypes = ['Deauthentication'] * int(num_frames * 0.95) + ['Beacon'] * int(num_frames * 0.05)
    else:
        # --- NORMAL PERIOD ---
        num_frames = normal_frames_per_second
        subtypes = random.choices(normal_subtypes, weights=normal_subtype_weights, k=num_frames)
    
    for subtype in subtypes:
        timestamps.append(second)
        frame_subtypes.append(subtype)

df_raw = pd.DataFrame({'timestamp': timestamps, 'subtype': frame_subtypes})
print(f"Generated {len(df_raw)} raw frame events over {total_duration_seconds} seconds.")


# ----------------------------------------
# 3. Feature Engineering: Time-Window Aggregation
# ----------------------------------------
print("\n--- Engineering Time-Series Features ---")

# We need to aggregate the raw frames into fixed time windows (1-second intervals)
# to create a consistent time-series dataset for our model.
df_agg = df_raw.groupby('timestamp')['subtype'].value_counts().unstack(fill_value=0)

# Engineer the most critical feature: the deauthentication ratio
df_agg['total_frames'] = df_agg.sum(axis=1)
if 'Deauthentication' not in df_agg.columns:
    df_agg['Deauthentication'] = 0 # Ensure the column exists even if no deauths were seen
    
df_agg['deauth_ratio'] = df_agg['Deauthentication'] / df_agg['total_frames']

# Select features for the model
features = ['total_frames', 'deauth_ratio', 'Beacon', 'Probe Request']
# Fill any missing columns that might not have appeared in a given second
for col in features:
    if col not in df_agg.columns:
        df_agg[col] = 0

df_model = df_agg[features].copy()

print("Aggregated data into 1-second windows. Sample:")
print(df_model.head())


# ----------------------------------------
# 4. Unsupervised Model Training
# ----------------------------------------
print("\n--- Unsupervised Model Training (on BENIGN data only) ---")

# We will train the model ONLY on the period before the attack starts.
# This teaches the model what "normal" Wi-Fi traffic looks like.
X_train_benign = df_model[df_model.index < attack_start_time]

print(f"Training Isolation Forest on {len(X_train_benign)} seconds of normal traffic data.")

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_benign)

# Initialize and train the Isolation Forest
model = IsolationForest(contamination='auto', random_state=42)
model.fit(X_train_scaled)
print("Training complete.")


# ----------------------------------------
# 5. Anomaly Detection and Evaluation
# ----------------------------------------
print("\n--- Detecting Anomalies on the Full Dataset ---")

# Now, use the trained model to get anomaly scores for the ENTIRE duration
X_all_scaled = scaler.transform(df_model)
df_model['anomaly_score'] = model.decision_function(X_all_scaled)
df_model['is_anomaly'] = model.predict(X_all_scaled) # -1 for anomaly, 1 for normal

# Create a ground truth label for comparison
df_model['ground_truth'] = np.where((df_model.index >= attack_start_time) & (df_model.index < attack_start_time + attack_duration), -1, 1)

print("\nPerformance Evaluation:")
# A simple accuracy check
accuracy = np.mean(df_model['is_anomaly'] == df_model['ground_truth'])
print(f"Accuracy in correctly identifying normal vs. attack periods: {accuracy:.2%}")


# ----------------------------------------
# 6. Visualization of Anomaly Detection
# ----------------------------------------
print("\n--- Visualizing the Detection Results ---")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

# Plot 1: The key feature - Deauthentication Ratio
ax1.plot(df_model.index, df_model['deauth_ratio'], label='Deauthentication Ratio', color='orange')
ax1.axvspan(attack_start_time, attack_start_time + attack_duration, color='red', alpha=0.2, label='Simulated Attack')
ax1.set_title('Feature: Deauthentication Ratio Over Time')
ax1.set_ylabel('Ratio')
ax1.legend()
ax1.grid(True)

# Plot 2: The model's anomaly score
ax2.plot(df_model.index, df_model['anomaly_score'], label='Anomaly Score', color='blue')
ax2.fill_between(df_model.index, plt.ylim()[0], plt.ylim()[1], where=df_model['is_anomaly']==-1,
                 facecolor='red', alpha=0.3, label='Detected Anomaly')
ax2.axvspan(attack_start_time, attack_start_time + attack_duration, color='red', alpha=0.2, label='Simulated Attack')
ax2.set_title('Isolation Forest Anomaly Score Over Time')
ax2.set_xlabel('Time (seconds)')
ax2.set_ylabel('Score')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()


# ----------------------------------------
# 7. Conclusion
# ----------------------------------------
print("\n--- Conclusion ---")
print("The Isolation Forest model successfully detected the deauthentication flood attack in the simulated Wi-Fi traffic.")
print("Key Takeaways:")
print("- The model, trained only on normal traffic, learned a baseline of behavior and correctly assigned a low anomaly score to it.")
print("- As soon as the attack began, the `deauth_ratio` feature spiked, causing a significant deviation from the baseline. The model immediately flagged this by producing a sharply negative anomaly score.")
print("- This demonstrates a powerful, signature-less approach to wireless intrusion detection. The system doesn't need to know the specifics of a 'deauthentication attack'; it only needs to know that a sudden, massive increase in deauth frames is not normal.")
print("- This method could be extended to detect other Wi-Fi anomalies, such as rogue access points (by monitoring for unusual beacon frames) or evil twin attacks (by watching for unusual association patterns).")
```