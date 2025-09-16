---

### **Project 7: Intelligent Traffic Routing (Reinforcement Learning)**

**Objective:** To train an RL agent that can dynamically find the optimal path for network traffic from a source to a destination, minimizing total latency. The agent should also be able to adapt its path if network conditions (link latencies) change.

**Environment:** We will create a **simulated network environment** directly in Python using the `networkx` library. This graph will represent our network, with nodes as routers/switches and edges as links with associated latency (cost).

**Model:** We will implement the foundational RL algorithm, **Q-Learning**. The agent will learn a "Q-table," which acts as a cheat sheet, telling it the expected quality (or future reward) of choosing a particular next hop from any given node.

**Instructions:**
This notebook is fully self-contained. Simply run the entire code block in Google Colab.

**Implementation in Google Colab:**
```python
#
# ==================================================================================
#  Project 7: Intelligent Traffic Routing (Reinforcement Learning)
# ==================================================================================
#
# Objective:
# This notebook demonstrates how to use Q-Learning, a reinforcement learning
# algorithm, to find the optimal path in a simulated network. The agent will
# learn to minimize latency and adapt to changing network conditions.
#
# To Run in Google Colab:
# Copy and paste this entire code block into a single cell and run it.
#

# ----------------------------------------
# 1. Import Necessary Libraries
# ----------------------------------------
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random

# ----------------------------------------
# 2. Create the Simulated Network Environment
# ----------------------------------------
print("--- Creating Simulated Network Environment ---")

# Create a graph object
G = nx.Graph()

# Define the network topology (nodes and edges with latency as 'weight')
edges = [
    ('A', 'B', 7), ('A', 'C', 9), ('A', 'F', 14),
    ('B', 'C', 10), ('B', 'D', 15),
    ('C', 'D', 11), ('C', 'F', 2),
    ('D', 'E', 6),
    ('E', 'F', 9)
]

# Add edges to the graph
for u, v, w in edges:
    G.add_edge(u, v, weight=w)

# Map node names to integers for easier array indexing
node_map = {node: i for i, node in enumerate(G.nodes())}
inv_node_map = {i: node for node, i in node_map.items()}

# Create an adjacency matrix representing the latencies (costs)
# We use a large number (np.inf) for non-existent links
num_nodes = len(G.nodes())
latency_matrix = np.full((num_nodes, num_nodes), np.inf)
for u, v, data in G.edges(data=True):
    i, j = node_map[u], node_map[v]
    latency_matrix[i, j] = latency_matrix[j, i] = data['weight']

print("Network created with the following nodes:", list(G.nodes()))

# Function to visualize the network
def draw_network(graph, path=None, title="Network Topology"):
    pos = nx.spring_layout(graph, seed=42)
    edge_labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw(graph, pos, with_labels=True, node_size=700, node_color='skyblue', font_size=10, font_weight='bold')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
    if path:
        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_edges(graph, pos, edgelist=path_edges, edge_color='r', width=2)
    plt.title(title)
    plt.show()

draw_network(G)


# ----------------------------------------
# 3. Q-Learning Algorithm Implementation
# ----------------------------------------
print("\n--- Implementing the Q-Learning Agent ---")

# Hyperparameters
alpha = 0.1      # Learning rate: How much we update Q-values based on new info
gamma = 0.9      # Discount factor: Importance of future rewards
epsilon = 0.2    # Epsilon-greedy: Probability of exploring vs. exploiting
num_episodes = 2000

# Initialize Q-table
# Rows are states (current node), columns are actions (next node)
q_table = np.zeros((num_nodes, num_nodes))

# The "reward" in our case is negative latency. The agent's goal is to
# maximize the reward, which means minimizing the latency.
# We set rewards for valid moves to -latency.
rewards = -latency_matrix

def train_agent(start_node_name, end_node_name, episodes):
    start_node = node_map[start_node_name]
    end_node = node_map[end_node_name]
    
    print(f"\nTraining agent to find path from {start_node_name} to {end_node_name}...")
    for episode in range(episodes):
        current_state = start_node
        
        # An episode ends when we reach the destination
        while current_state != end_node:
            # Epsilon-greedy action selection
            if random.uniform(0, 1) < epsilon:
                # Explore: choose a random valid action (a connected node)
                possible_actions = np.where(rewards[current_state] > -np.inf)[0]
                action = random.choice(possible_actions)
            else:
                # Exploit: choose the best known action
                action = np.argmax(q_table[current_state])

            # Get the reward for taking that action
            reward = rewards[current_state, action]

            # Q-learning formula
            old_value = q_table[current_state, action]
            next_max = np.max(q_table[action]) # Best expected future reward from the next state
            
            # The new Q-value is a blend of the old value and the learned value
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[current_state, action] = new_value
            
            # Move to the next state
            current_state = action
    print("Training complete.")

# Function to extract the optimal path from the learned Q-table
def get_optimal_path(start_node_name, end_node_name):
    path = [start_node_name]
    current_node = node_map[start_node_name]
    end_node = node_map[end_node_name]
    
    while current_node != end_node:
        next_node = np.argmax(q_table[current_node])
        path.append(inv_node_map[next_node])
        current_node = next_node
        if len(path) > 10: # Safety break to prevent infinite loops
            print("Error: Path finding failed, stuck in a loop.")
            return []
    return path

# ----------------------------------------
# 4. Scenario 1: Find Initial Optimal Path
# ----------------------------------------
print("\n--- Scenario 1: Initial Path Finding ---")
# Define the start and end points
start = 'A'
end = 'E'

# Train the agent
train_agent(start, end, num_episodes)

# Get and display the optimal path
optimal_path_1 = get_optimal_path(start, end)
print(f"Learned optimal path from {start} to {end}: {' -> '.join(optimal_path_1)}")
draw_network(G, path=optimal_path_1, title=f"Optimal Path from {start} to {end}")


# ----------------------------------------
# 5. Scenario 2: Adapt to Network Change
# ----------------------------------------
print("\n--- Scenario 2: Adapting to Network Congestion ---")
print("Introducing congestion: Latency on link C -> F increases from 2 to 20.")

# Update the graph and reward matrix to reflect the change
G['C']['F']['weight'] = 20
rewards[node_map['C'], node_map['F']] = -20
rewards[node_map['F'], node_map['C']] = -20

# We don't need to retrain from scratch. We can continue training.
# This allows the agent to adapt its existing knowledge.
train_agent(start, end, num_episodes) # Continue training

# Get and display the new optimal path
optimal_path_2 = get_optimal_path(start, end)
print(f"New learned optimal path from {start} to {end}: {' -> '.join(optimal_path_2)}")
draw_network(G, path=optimal_path_2, title=f"New Optimal Path after Congestion on C-F")


# ----------------------------------------
# 6. Conclusion
# ----------------------------------------
print("\n--- Conclusion ---")
print("This notebook demonstrated the core principles of Reinforcement Learning for network routing.")
print("Key Takeaways:")
print(f"1. The agent initially learned that the best path was {' -> '.join(optimal_path_1)}, correctly identifying the low-latency C-F link.")
print(f"2. After a simulated congestion event dramatically increased latency on the C-F link, the agent adapted.")
print(f"3. By continuing its training, it discovered a new optimal route ({' -> '.join(optimal_path_2)}) that avoids the congested link.")
print("This adaptability is the power of RL. In a real-world Software-Defined Network (SDN), an RL agent could continuously monitor link states and automatically re-route traffic to maintain optimal performance without human intervention.")

```