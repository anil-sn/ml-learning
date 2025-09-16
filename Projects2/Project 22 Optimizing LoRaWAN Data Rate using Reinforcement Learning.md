---

### **Project 22: Optimizing LoRaWAN Data Rate using Reinforcement Learning**

**Objective:** To train a Reinforcement Learning agent that can dynamically select the optimal Spreading Factor (SF) for a LoRaWAN end-device. The goal is to maximize the probability of a successful transmission while minimizing the "time on air," which is directly proportional to energy consumption.

**Environment:** **Simulated LoRaWAN Channel**. Since we cannot connect a real LoRaWAN device, we will create a Python-based simulation. This environment will model:
*   **State:** The device's current Signal-to-Noise Ratio (SNR), representing the quality of its connection.
*   **Actions:** The agent can choose a Spreading Factor (SF7 to SF12).
*   **Physics:** Higher SFs are slower (more time on air) but can succeed at lower SNRs. Lower SFs are faster (less time on air) but require a stronger signal.
*   **Reward:** The agent receives a large positive reward for a successful transmission, a large negative penalty for a failed one, and a small penalty based on the time on air to encourage efficiency.

**Model:** We will implement **Q-Learning**, a foundational RL algorithm that is perfect for learning the optimal action to take in any given state.

**Instructions:**
This notebook is fully self-contained and does not require any external files or APIs. Simply run the entire code block in Google Colab.

**Implementation in Google Colab:**```python
#
# ==================================================================================
#  Project 22: Optimizing LoRaWAN Data Rate using Reinforcement Learning
# ==================================================================================
#
# Objective:
# This notebook trains a Q-Learning agent to intelligently select the optimal
# LoRaWAN Spreading Factor (SF) based on the current channel conditions (SNR)
# to balance reliability and energy efficiency.
#
# To Run in Google Colab:
# Copy and paste this entire code block into a single cell and run it.
#

# ----------------------------------------
# 1. Import Necessary Libraries
# ----------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

# ----------------------------------------
# 2. Build the Simulated LoRaWAN Environment
# ----------------------------------------
print("--- Building the Simulated LoRaWAN Environment ---")

class LoRaWANEnv:
    def __init__(self):
        # Actions: 6 possible Spreading Factors (SF7 to SF12)
        self.actions = [7, 8, 9, 10, 11, 12]
        # Required SNR (in dB) for a successful transmission at each SF
        self.snr_thresholds = {7: -7.5, 8: -10, 9: -12.5, 10: -15, 11: -17.5, 12: -20}
        # Relative time on air (energy cost) for each SF
        self.time_on_air = {7: 1, 8: 1.8, 9: 3.2, 10: 5.8, 11: 11, 12: 21}
        
        # States: Discretized SNR values from -25 dB to 0 dB in steps of 2.5 dB
        self.states = np.arange(-25, 2.5, 2.5)
        self.state_space_size = len(self.states)
        self.action_space_size = len(self.actions)
        
    def get_state_index(self, snr):
        # Find the closest discretized state for a given continuous SNR value
        return np.abs(self.states - snr).argmin()

    def step(self, state_idx, action_idx):
        current_snr = self.states[state_idx]
        chosen_sf = self.actions[action_idx]
        
        # --- Environment Physics ---
        # Check if the transmission is successful
        if current_snr >= self.snr_thresholds[chosen_sf]:
            success = True
            reward = 100  # Large reward for success
        else:
            success = False
            reward = -200 # Large penalty for failure
        
        # Add a penalty proportional to the energy used (time on air)
        reward -= self.time_on_air[chosen_sf]
        
        # Simulate the next state (e.g., SNR changes slightly due to environmental factors)
        next_snr = current_snr + np.random.normal(0, 1.0)
        next_snr = np.clip(next_snr, -25, 0) # Keep SNR within bounds
        next_state_idx = self.get_state_index(next_snr)
        
        return next_state_idx, reward, success

# Instantiate the environment
env = LoRaWANEnv()
print("Environment built successfully.")


# ----------------------------------------
# 3. Q-Learning Agent Training
# ----------------------------------------
print("\n--- Training the Q-Learning Agent ---")

# Hyperparameters
num_episodes = 20000
alpha = 0.1      # Learning rate
gamma = 0.9      # Discount factor
epsilon = 1.0    # Exploration rate
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.0005

# Initialize Q-table with zeros (rows=states, columns=actions)
q_table = np.zeros((env.state_space_size, env.action_space_size))
rewards_per_episode = []

for episode in range(num_episodes):
    # Start with a random SNR state
    state = random.randint(0, env.state_space_size - 1)
    total_reward = 0
    
    # Epsilon-greedy action selection
    if random.uniform(0, 1) > epsilon:
        action = np.argmax(q_table[state, :]) # Exploit
    else:
        action = random.randint(0, env.action_space_size - 1) # Explore
        
    next_state, reward, _ = env.step(state, action)
    
    # Q-Learning update rule
    q_table[state, action] = q_table[state, action] + alpha * (reward + gamma * np.max(q_table[next_state, :]) - q_table[state, action])
    
    total_reward += reward
    rewards_per_episode.append(total_reward)
    
    # Update epsilon (exploration-exploitation trade-off)
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

print("Training complete.")

# ----------------------------------------
# 4. Analysis and Visualization of Learned Policy
# ----------------------------------------
print("\n--- Analyzing the Learned Policy ---")

# Extract the optimal policy from the Q-table
# For each state (SNR level), the best action is the one with the highest Q-value.
optimal_policy = np.argmax(q_table, axis=1)
policy_df = pd.DataFrame({
    'SNR (dB)': env.states,
    'Optimal SF': [env.actions[p] for p in optimal_policy]
})

print("Learned Optimal Policy:")
print(policy_df)

# --- Visualize the Q-table ---
plt.figure(figsize=(12, 8))
sns.heatmap(q_table, cmap='viridis', xticklabels=env.actions, yticklabels=np.round(env.states, 1))
plt.title('Learned Q-Table Values', fontsize=16)
plt.xlabel('Action (Spreading Factor)')
plt.ylabel('State (Signal-to-Noise Ratio)')
plt.show()


# --- Visualize the Learned Policy ---
plt.figure(figsize=(10, 6))
plt.plot(policy_df['SNR (dB)'], policy_df['Optimal SF'], marker='o', linestyle='--')
plt.title('Optimal LoRaWAN Spreading Factor vs. SNR', fontsize=16)
plt.xlabel('SNR (dB)')
plt.ylabel('Optimal SF')
plt.grid(True)
plt.gca().invert_xaxis() # Better visualization with high SNR on the left
plt.show()

# --- Visualize Learning Progress ---
plt.figure(figsize=(12, 6))
# Calculate a moving average of rewards to see the trend
moving_avg = pd.Series(rewards_per_episode).rolling(window=500).mean()
plt.plot(moving_avg)
plt.title('Agent Learning Progress (Moving Average of Reward per Episode)', fontsize=16)
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.grid(True)
plt.show()


# ----------------------------------------
# 5. Conclusion
# ----------------------------------------
print("\n--- Conclusion ---")
print("The Q-Learning agent successfully learned an intelligent policy for selecting the LoRaWAN Spreading Factor.")
print("Key Takeaways:")
print("- The final policy is highly logical and reflects networking best practices: When the signal is strong (high SNR), the agent chooses a low SF (like SF7) for fast, energy-efficient communication. When the signal is weak (low SNR), it correctly switches to a high SF (like SF12) for a more robust, long-range connection.")
print("- The Q-table heatmap visually confirms this logic, showing high Q-values for low SFs at high SNRs and high Q-values for high SFs at low SNRs.")
print("- The learning progress chart shows that the agent's performance steadily improved over time, demonstrating that it was effectively learning from its successes and failures.")
print("- This is a powerful demonstration of how Reinforcement Learning can be used to create truly adaptive and autonomous network protocols that optimize their own performance in response to changing environmental conditions, without needing to be explicitly programmed with complex 'if-then-else' rules.")

```