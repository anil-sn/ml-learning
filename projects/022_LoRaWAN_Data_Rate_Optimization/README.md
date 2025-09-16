# Project 22: Optimizing LoRaWAN Data Rate using Reinforcement Learning

## Objective
Train a Reinforcement Learning agent that can dynamically select the optimal Spreading Factor (SF) for a LoRaWAN end-device to maximize successful transmission probability while minimizing energy consumption (time on air).

## Business Value
- **Energy Efficiency**: Extend battery life of IoT devices by optimizing transmission parameters
- **Network Performance**: Improve overall network throughput and reliability
- **Dynamic Optimization**: Automatically adapt to changing channel conditions without manual intervention
- **Cost Reduction**: Reduce operational costs through intelligent power management
- **Scalability**: Enable autonomous operation of large-scale IoT deployments

## Core Libraries
- **numpy**: Numerical computing and Q-table operations
- **pandas**: Data manipulation and policy analysis
- **matplotlib & seaborn**: Visualization of learning progress and policy
- **Q-Learning**: Model-free reinforcement learning algorithm for decision making

## Dataset
**Source**: Simulated LoRaWAN Environment
- **State Space**: Discretized SNR values from -25 dB to 0 dB (environmental conditions)
- **Action Space**: Spreading Factors SF7 to SF12 (transmission parameters)
- **Physics Model**: SNR thresholds and time-on-air relationships for each SF
- **Reward Structure**: Success/failure rewards with energy consumption penalties

## Step-by-Step Guide

### 1. Environment Setup
```python
# No external dependencies required - fully self-contained simulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
```

### 2. LoRaWAN Environment Simulation
```python
class LoRaWANEnv:
    def __init__(self):
        # Spreading Factor options (SF7 to SF12)
        self.actions = [7, 8, 9, 10, 11, 12]
        
        # Required SNR thresholds for successful transmission
        self.snr_thresholds = {
            7: -7.5, 8: -10, 9: -12.5, 
            10: -15, 11: -17.5, 12: -20
        }
        
        # Relative energy consumption (time on air)
        self.time_on_air = {
            7: 1, 8: 1.8, 9: 3.2, 
            10: 5.8, 11: 11, 12: 21
        }
        
        # Discretized SNR states
        self.states = np.arange(-25, 2.5, 2.5)
```

### 3. Q-Learning Algorithm Implementation
```python
# Initialize Q-table with zeros
q_table = np.zeros((env.state_space_size, env.action_space_size))

# Hyperparameters
num_episodes = 20000
alpha = 0.1      # Learning rate
gamma = 0.9      # Discount factor
epsilon = 1.0    # Initial exploration rate
decay_rate = 0.0005  # Epsilon decay
```

### 4. Agent Training Loop
```python
for episode in range(num_episodes):
    # Start with random initial state
    state = random.randint(0, env.state_space_size - 1)
    
    # Epsilon-greedy action selection
    if random.uniform(0, 1) > epsilon:
        action = np.argmax(q_table[state, :])  # Exploit
    else:
        action = random.randint(0, env.action_space_size - 1)  # Explore
    
    # Execute action and observe reward
    next_state, reward, success = env.step(state, action)
    
    # Q-Learning update rule
    q_table[state, action] = q_table[state, action] + alpha * (
        reward + gamma * np.max(q_table[next_state, :]) - q_table[state, action]
    )
    
    # Decay exploration rate
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
```

### 5. Policy Extraction and Analysis
```python
# Extract optimal policy from Q-table
optimal_policy = np.argmax(q_table, axis=1)
policy_df = pd.DataFrame({
    'SNR (dB)': env.states,
    'Optimal SF': [env.actions[p] for p in optimal_policy]
})

# Visualize learned Q-values
sns.heatmap(q_table, cmap='viridis', 
            xticklabels=env.actions, 
            yticklabels=np.round(env.states, 1))
plt.title('Learned Q-Table Values')
plt.show()
```

### 6. Performance Evaluation
```python
# Plot optimal policy
plt.plot(policy_df['SNR (dB)'], policy_df['Optimal SF'], 
         marker='o', linestyle='--')
plt.title('Optimal LoRaWAN Spreading Factor vs. SNR')
plt.xlabel('SNR (dB)')
plt.ylabel('Optimal SF')
plt.gca().invert_xaxis()  # High SNR on left
plt.show()

# Learning progress visualization
moving_avg = pd.Series(rewards_per_episode).rolling(window=500).mean()
plt.plot(moving_avg)
plt.title('Agent Learning Progress')
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.show()
```

### 7. Real-World Deployment Function
```python
def select_optimal_sf(current_snr, q_table, env):
    """
    Select optimal Spreading Factor for current channel conditions
    
    Args:
        current_snr: Current signal-to-noise ratio in dB
        q_table: Trained Q-table
        env: LoRaWAN environment
    
    Returns:
        Optimal spreading factor (SF7-SF12)
    """
    state_idx = env.get_state_index(current_snr)
    action_idx = np.argmax(q_table[state_idx, :])
    return env.actions[action_idx]
```

## Success Criteria
- **Primary Metric**: Average reward per episode increases over training
- **Policy Quality**: Learned policy follows expected SNR-SF relationship
- **Convergence**: Q-values stabilize after sufficient training episodes
- **Energy Efficiency**: Balance between success rate and energy consumption
- **Adaptability**: Agent learns to handle varying channel conditions

## Next Steps & Extensions

### Technical Enhancements
1. **Deep Q-Learning**: Replace tabular Q-learning with neural networks for continuous states
2. **Multi-Agent Systems**: Coordinate multiple LoRaWAN devices to avoid interference
3. **Advanced Algorithms**: Implement Actor-Critic, PPO, or other modern RL methods
4. **Real Hardware Integration**: Connect to actual LoRaWAN transceivers for validation

### Business Applications
1. **Smart City IoT**: Optimize thousands of sensors for environmental monitoring
2. **Agricultural IoT**: Maximize battery life for remote field sensors
3. **Industrial IoT**: Ensure reliable communication in harsh manufacturing environments
4. **Asset Tracking**: Balance location update frequency with battery consumption

### Research Directions
1. **Transfer Learning**: Adapt policies learned in one environment to another
2. **Federated Learning**: Train policies across distributed LoRaWAN networks
3. **Multi-Objective Optimization**: Consider latency, reliability, and energy simultaneously
4. **Uncertainty Modeling**: Handle unknown or varying channel conditions

## Files in this Project
- `README.md` - Project documentation and implementation guide
- `lorawan_data_rate_optimization.ipynb` - Complete Jupyter notebook implementation
- `requirements.txt` - Python package dependencies

## Key Insights
- Q-Learning successfully discovers the optimal SNR-to-SF mapping without explicit programming
- The learned policy demonstrates networking best practices: low SF for strong signals, high SF for weak signals
- Reinforcement Learning enables truly adaptive protocols that respond to environmental changes
- The approach balances transmission success with energy efficiency automatically
- Visualization of Q-values and learning progress provides interpretable insights into agent behavior

## LoRaWAN Physics Model
- **Spreading Factors**: SF7 (fastest, least energy) to SF12 (slowest, most energy)
- **SNR Requirements**: Higher SFs can operate at lower SNR levels
- **Energy Trade-off**: Time on air increases exponentially with higher SFs
- **Success Probability**: Determined by SNR meeting the required threshold for chosen SF
- **Environmental Dynamics**: SNR varies over time due to interference and propagation conditions