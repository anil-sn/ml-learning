
### **Project 10: QoS Optimization Using Reinforcement Learning**

#### **1. Objective**
To build a Reinforcement Learning (RL) agent that learns an optimal policy for dynamically managing network resources to maintain Quality of Service (QoS). Specifically, the agent will decide whether to **accept** or **reject** incoming high-bandwidth service requests based on the current network state to prevent congestion and latency spikes.

#### **2. Business Value**
An intelligent QoS management system can significantly enhance network value:
*   **Maximize Revenue:** By intelligently accepting high-value traffic without compromising the performance for existing users, it maximizes the network's utility.
*   **Guarantee SLAs:** Automatically enforces policies to ensure that Service Level Agreements for latency-sensitive applications (like VoIP or streaming) are met.
*   **Automate Network Operations:** Replaces static, rule-based QoS policies (which require manual tuning) with a dynamic, self-learning system that adapts to changing traffic patterns.

#### **3. Core Libraries**
*   `gymnasium`: The modern fork of OpenAI's Gym, providing the standard toolkit for building RL environments.
*   `stable-baselines3`: A set of reliable implementations of state-of-the-art RL algorithms, built on PyTorch.
*   `numpy`: For managing state and action spaces.
*   `matplotlib`: To visualize the agent's performance during and after training.

#### **4. Dataset**
*   **Approach:** **A Custom Simulated Environment**.
*   **Why:** Reinforcement Learning requires an interactive environment, not a static dataset. The RL agent learns by taking actions and observing the consequences (rewards). We will create a simplified simulation of a network link that our agent can control. This is a standard and necessary approach for applying RL to real-world control problems.

#### **5. Detailed Step-by-Step Guide**

**Step 1: Setup the Environment**
1.  Create a project folder and a Python virtual environment.
    ```bash
    mkdir rl-qos-optimizer
    cd rl-qos-optimizer
    python -m venv venv
    source venv/bin/activate
    ```
2.  Install the necessary libraries. `stable-baselines3` requires PyTorch.
    ```bash
    pip install gymnasium stable-baselines3[extra] torch numpy matplotlib jupyterlab
    ```
3.  Start a Jupyter Lab session for building and testing the environment and agent.

**Step 2: Design and Build the Custom Gym Environment**
This is the most critical part of the project. We will create a Python class that inherits from `gym.Env` and defines our network simulation.
1.  Create a file named `network_env.py`.
2.  Inside this file, define the environment class:
    ```python
    import gymnasium as gym
    import numpy as np

    class NetworkQoSEnv(gym.Env):
        def __init__(self):
            super(NetworkQoSEnv, self).__init__()

            # Define the capacity of our network link (e.g., 100 Gbps)
            self.link_capacity = 100
            
            # --- Define Action Space ---
            # Action 0: Reject the incoming service request
            # Action 1: Accept the incoming service request
            self.action_space = gym.spaces.Discrete(2)

            # --- Define Observation Space (the state) ---
            # The state is represented by two values:
            # 1. Current bandwidth usage on the link (0 to capacity)
            # 2. Bandwidth of the new service request (e.g., 5 to 30 Gbps)
            self.observation_space = gym.spaces.Box(low=0, high=self.link_capacity, shape=(2,), dtype=np.float32)

            # --- Initialize State ---
            self.current_usage = 0
            self.new_request = 0

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            # Reset the network to an idle state at the beginning of an episode
            self.current_usage = np.random.uniform(20, 50) # Start with some base traffic
            self.new_request = np.random.uniform(5, 30)
            return np.array([self.current_usage, self.new_request], dtype=np.float32), {}

        def step(self, action):
            # --- Define the core logic of the environment ---
            if action == 1: # Agent chose to ACCEPT the request
                self.current_usage += self.new_request
            
            # --- Define the Reward Function ---
            # This is where we teach the agent what is "good" or "bad"
            if self.current_usage > self.link_capacity:
                # Heavy penalty for exceeding capacity (congestion)
                reward = -100
                terminated = True # End the episode on failure
            elif action == 1: # Accepted successfully
                # Reward is proportional to the bandwidth of the accepted request
                reward = self.new_request / 10
                terminated = False
            else: # action == 0 (Rejected)
                # Small penalty for rejecting a potentially valuable request
                reward = -1
                terminated = False

            # --- Prepare for the next step ---
            # Simulate some traffic leaving the network
            self.current_usage *= np.random.uniform(0.8, 0.95)
            # Generate a new service request for the next state
            self.new_request = np.random.uniform(5, 30)
            
            # The episode ends if the agent causes congestion
            if self.current_usage > self.link_capacity:
                terminated = True

            return np.array([self.current_usage, self.new_request], dtype=np.float32), reward, terminated, False, {}
    ```

**Step 3: Test the Environment**
Before training, it's vital to ensure the environment behaves as expected.
1.  In your Jupyter Notebook, import and test the environment.
    ```python
    from network_env import NetworkQoSEnv
    from stable_baselines3.common.env_checker import check_env

    env = NetworkQoSEnv()
    # This utility checks if your environment follows the Gym API
    check_env(env)
    print("Environment check passed!")
    ```

**Step 4: Train the Reinforcement Learning Agent**
We will use the **PPO (Proximal Policy Optimization)** algorithm from Stable Baselines3, which is a robust and widely used algorithm.
```python
from stable_baselines3 import PPO

# Re-create the environment
env = NetworkQoSEnv()

# Instantiate the PPO agent with a Multi-Layer Perceptron policy
model = PPO("MlpPolicy", env, verbose=1)

# Train the agent for 20,000 steps
model.train(total_timesteps=20000)
print("\nTraining complete!")
```
*You will see output showing the agent's `mean_reward` gradually increasing as it learns the optimal policy.*

**Step 5: Evaluate the Trained Agent**
Let's see how our trained agent behaves in the environment.
```python
obs, _ = env.reset()
total_reward = 0
num_steps = 100

for step in range(num_steps):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    action_str = "Accept" if action == 1 else "Reject"
    print(f"Step {step+1}: State=({obs[0]:.1f}, {obs[1]:.1f}) -> Action={action_str}, Reward={reward:.1f}")

    if terminated:
        print("Episode terminated (Congestion).")
        break

print(f"\nTotal reward over {num_steps} steps: {total_reward:.2f}")
```
*   **Analyze the output:** You should observe that the agent has learned a smart policy. It will **accept** requests when the `current_usage` is low but will start **rejecting** them as `current_usage` gets close to the `link_capacity` (100) to avoid the heavy penalty.

**Step 6: Compare with a Simple, Rule-Based Policy**
To prove the value of the RL agent, let's compare its performance to a simple, "greedy" policy that accepts every request until the capacity is full.
```python
def greedy_policy(state):
    # Rule: Accept if there is enough capacity, otherwise reject.
    current_usage, new_request = state
    if current_usage + new_request <= 100:
        return 1 # Accept
    else:
        return 0 # Reject

obs, _ = env.reset()
total_reward_greedy = 0
for step in range(num_steps):
    action = greedy_policy(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward_greedy += reward
    if terminated: break

print(f"Greedy Policy Total Reward: {total_reward_greedy:.2f}")
print(f"RL Agent Total Reward: {total_reward:.2f}")
```
*You will likely find that the RL agent achieves a higher total reward because it learns a more nuanced policy than the simple rule-based approach.*

#### **6. Success Criteria**
*   The team can successfully design and implement a custom `gymnasium` environment that simulates the network QoS problem.
*   The `stable-baselines3` PPO agent is successfully trained on the custom environment, and the team can observe the `mean_reward` increasing during training.
*   The team can demonstrate through evaluation that the trained agent has learned a sensible policy (i.e., it rejects traffic appropriately to avoid congestion).
*   The RL agent's performance (in terms of total reward) is shown to be superior to a simple, hard-coded greedy policy.

#### **7. Next Steps & Extensions**
*   **More Complex State:** Enhance the environment by adding more dimensions to the state, such as the *type* of service request (e.g., low-latency VoIP vs. high-throughput file transfer).
*   **More Complex Action Space:** Instead of a simple accept/reject, the action space could be to assign the request to one of several queues with different priority levels.
*   **Sim-to-Real:** While a long-term goal, the principles learned here are the first step toward deploying such an agent in a real network. The next stage would be to test the agent in a more realistic network simulator (like GNS3 or ns-3) before considering live deployment.